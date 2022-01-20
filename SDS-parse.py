# %%

import json
import math
from typing import List
from datetime import datetime
from konlpy.tag import Okt
import pickle

okt = Okt()

# %%

class Base:
  def __init__(self, sessionId: str, userId: str, time: str, eventName: str):
    self.eventName = ''
    self.sessionId = sessionId
    self.userId = userId
    self.time = datetime.fromisoformat(time)

class Session:
  def __init__(self, sessionId: str, userId: str, metrics: dict, sequence: List[Base]):
    self.sessionId = sessionId
    self.userId = userId
    self.metrics = metrics
    self.sequence = sequence
    self.BERTopicKeywordCluster = -1
    self.ClusterID = -1

class Query(Base):
  def __init__(self, query: str, summary: dict, queryResults: List[dict], sessionId: str, userId: str, isRelated: bool, time: str, extendedQuery: str, previousQuery: str):
    self.query = query
    self.queryResults = queryResults
    self.summary = summary
    self.isRelated = isRelated
    self.extendedQuery = extendedQuery
    self.isRefined = isReformulated(query, previousQuery)
    super().__init__(sessionId = sessionId, userId = userId, time = time, eventName = 'Query')

  def setIntermediateMetrics(self, metrics: dict):
    self.intermediateMetrics = metrics


class Click(Base):
  def __init__(self, data: dict, sessionId: str, userId: str, query: str):
    self.query = query
    self.target_doc_url = data['argument']['target_doc_url']
    self.target_doc_seq = data['argument']['target_doc_seq']
    self.target_doc_title = data['argument']['target_doc_title']
    self.target_doc_context = data['argument']['target_doc_context']
    self.stay_time_sec = data['stay_time_sec']

    super().__init__(sessionId = sessionId, userId = userId, time = data['timestamp'], eventName='Click')

class ClickQuickLink(Base):
  def __init__(self, data: dict, sessionId: str, userId: str, query: str):
    self.query = query
    self.req = data['argument']['req']
    self.match_keyword = data['argument']['match_keyword']
    self.source = data['argument']['source']
    self.title = data['argument']['title']

    super().__init__(sessionId = sessionId, userId = userId, time = data['timestamp'], eventName='ClickQuickLink')

class EndSession(Base):

  def __init__(self, data: dict, sessionId: str, userId: str, query: str):
    self.query = query
    self.by = data['argument']['by']
    super().__init__(sessionId = sessionId, userId = userId, time = data['timestamp'], eventName='EndSession')

class Session:
  def __init__(self, sessionId: str, userId: str, sequence: List[Base], metrics: dict):
    self.sessionId = sessionId
    self.userId = userId
    self.sequence = sequence
    self.metrics = metrics

# %%
def isReformulated(query: str, prev_query: str):

  if prev_query == None:
    return False

  if jamo_levenshtein(query, prev_query) < 3:
    return True
  
  query_nouns = set(okt.nouns(query))
  prev_query_nouns = set(okt.nouns(prev_query))

  union = query_nouns.union(prev_query_nouns)
  intersect = query_nouns.intersect(prev_query_nouns)

  return len(intersect) / len(union) > 0.5

def process_action(action: dict, sessionId: str, userId: str, query: str):
  if action['action_type'] == 'select_doc':
    return Click(data = action, sessionId = sessionId, userId = userId, query = query)
  elif action['action_type'] == 'select_item' and action['item_name'] == 'quick_link':
    return ClickQuickLink(data = action, sessionId = sessionId, userId = userId, query = query)
  elif action['action_type'] == 'select_item' and action['item_name'] == 'rel_search_keyword':
    pass
  elif action['action_type'] == 'close_session':
    return EndSession(data = action, sessionId = sessionId, userId = userId, query = query)
  else:
    raise Exception(f'Unknown action type: {action["action_type"]}')


# def compute_stats(session: List[Base]):

#   total_queries = [e for e in session if isinstance(e, Query)]

#   total_query_count = len(total_queries)

#   abandoned_query_count = len([q for q in total_queries if (q.summary['is_select_doc'] == False and q.summary['is_select_top5_doc'] == False)])

def levenshtein(s1, s2, cost=None, debug=False): 
  # adopted from https://lovit.github.io/nlp/2018/08/28/levenshtein_hangle/
  if len(s1) < len(s2):
    return levenshtein(s2, s1, debug=debug)

  if len(s2) == 0:
    return len(s1)

  if cost is None:
    cost = {}

  # changed
  def substitution_cost(c1, c2):
    if c1 == c2:
      return 0
    return cost.get((c1, c2), 1)

  previous_row = range(len(s2) + 1)
  for i, c1 in enumerate(s1):
    current_row = [i + 1]
    for j, c2 in enumerate(s2):
      insertions = previous_row[j + 1] + 1
      deletions = current_row[j] + 1
      # Changed
      substitutions = previous_row[j] + substitution_cost(c1, c2)
      current_row.append(min(insertions, deletions, substitutions))

    if debug:
      print(current_row[1:])

    previous_row = current_row

  return previous_row[-1]

kor_begin = 44032
kor_end = 55203
chosung_base = 588
jungsung_base = 28
jaum_begin = 12593
jaum_end = 12622
moum_begin = 12623
moum_end = 12643

chosung_list = [ 'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 
  'ㅅ', 'ㅆ', 'ㅇ' , 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 
  'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 
  'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 
  'ㅡ', 'ㅢ', 'ㅣ']

jongsung_list = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ',
  'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 
  'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 
  'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jaum_list = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ', 
  'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 
  'ㅃ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

moum_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 
  'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

def compose(chosung, jungsung, jongsung):
  char = chr(
    kor_begin +
    chosung_base * chosung_list.index(chosung) +
    jungsung_base * jungsung_list.index(jungsung) +
    jongsung_list.index(jongsung)
  )
  return char

def decompose(c):
  if not character_is_korean(c):
    return None
  i = ord(c)
  if (jaum_begin <= i <= jaum_end):
    return (c, ' ', ' ')
  if (moum_begin <= i <= moum_end):
    return (' ', c, ' ')

  # decomposition rule
  i -= kor_begin
  cho  = i // chosung_base
  jung = ( i - cho * chosung_base ) // jungsung_base 
  jong = ( i - cho * chosung_base - jung * jungsung_base )    
  return (chosung_list[cho], jungsung_list[jung], jongsung_list[jong])

def character_is_korean(c):
    i = ord(c)
    return ((kor_begin <= i <= kor_end) or
      (jaum_begin <= i <= jaum_end) or
      (moum_begin <= i <= moum_end))

def jamo_levenshtein(s1, s2, debug=False):
  if len(s1) < len(s2):
    return jamo_levenshtein(s2, s1, debug)

  if len(s2) == 0:
    return len(s1)

  def substitution_cost(c1, c2):
    if c1 == c2:
        return 0
    return levenshtein(decompose(c1), decompose(c2))/3

  previous_row = range(len(s2) + 1)
  for i, c1 in enumerate(s1):
    current_row = [i + 1]
    for j, c2 in enumerate(s2):
      insertions = previous_row[j + 1] + 1
      deletions = current_row[j] + 1
      # Changed
      substitutions = previous_row[j] + substitution_cost(c1, c2)
      current_row.append(min(insertions, deletions, substitutions))

    if debug:
      print(['%.3f'%v for v in current_row[1:]])

    previous_row = current_row

  return previous_row[-1]

def compute_intermediate_metrics(queryEvent: Query, clickEvents: List[Base]):
  reciprocal_ranks = [(1 / c.target_doc_seq) for c in clickEvents if isinstance(c, Click)]

  meanRR = (sum(reciprocal_ranks) / len(reciprocal_ranks)) if len(reciprocal_ranks) > 0 else 0
  maxRR = max(reciprocal_ranks) if len(reciprocal_ranks) > 0 else 0

  isAbandoned = queryEvent.summary['is_select_doc'] == False and queryEvent.summary['is_select_top5_doc'] == False

  isClickTop5 = queryEvent.summary['is_select_top5_doc']

  return {
    meanRR: meanRR,
    maxRR: maxRR,
    isAbandoned: isAbandoned,
    isClickTop5: isClickTop5
  }

def compute_stats(session: List[Base]):
  queryEvents = [e for e in session if isinstance(e, Query)]
  totalQueryCount = len(queryEvents)

  meanRR = sum([q.intermediateMetrics['meanRR'] for q in queryEvents if q.intermediateMetrics['meanRR'] > 0]) / len([q.intermediateMetrics['meanRR'] for q in queryEvents if q.intermediateMetrics['meanRR'] > 0])

  maxRR = sum([q.intermediateMetrics['maxRR'] for q in queryEvents if q.intermediateMetrics['maxRR'] > 0]) / len([q.intermediateMetrics['maxRR'] for q in queryEvents if q.intermediateMetrics['maxRR'] > 0])
  
  abandonmentRate = len([q for q in queryEvents if q.intermediateMetrics['isAbandoned']]) / totalQueryCount
  reformualtionRate = len([q for q in queryEvents if q.intermediateMetrics['isReformulated']]) / totalQueryCount
  clickTop5Rate = len([q for q in queryEvents if q.intermediateMetrics['isClickTop5']]) / totalQueryCount

  return {
    meanRR: meanRR,
    maxRR: maxRR,
    abandonmentRate: abandonmentRate,
    reformualtionRate: reformualtionRate,
    clickTop5Rate: clickTop5Rate
  }

  

# %%

with open('./SDS-log-sample.json', 'r') as f:
  data = json.load(f)

hits = data['hits']['hits']

sessionIds = set([h['_source']['session_id'] for h in hits]) # extract unique session ids

sessions_total = []

for sessionId in sessionIds:
  entries = [h for h in hits if h['_source']['session_id'] == sessionId]

  events = []
  userId = entries[0]['_source']['user_id']


  for entry in entries:
    # Extracting events from the log
    query = entry['_source']['query']['keyword']
    queryEvent = Query(
      query = entry['_source']['query']['keyword'],
      queryResults = entry['_source']['doc_result']['documents'],
      sessionId = sessionId,
      userId = userId,
      summary = entry['_source']['summary'],
      time = entry['_source']['query']['timestamp'],
      isRelated = entry['_source']['query']['from'] is not None,
      extendedQuery = entry['_source']['query']['extend_keyword']
    )
    events.append(queryEvent)

    clickEvents = [process_action(a, sessionId, userId, query) for a in entry['_source']['actions']]

    events.extend(clickEvents)

    intermediateStats = compute_intermediate_metrics(queryEvent = queryEvent, clickEvents = clickEvents)
    queryEvent.setIntermediateMetrics(intermediateStats)

  sorted_events = sorted(events, key = lambda x: x.time)
  session_metrics = compute_stats(sorted_events)

  new_session = Session(
    sessionId = sessionId,
    userId=userId,
    metrics=session_metrics,
    sequence = sorted_events
  )

  sessions_total.append(new_session)

  # store them into the keyword_cluster format

  # process events array to make sth







# %%
