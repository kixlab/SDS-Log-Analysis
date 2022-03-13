# %%
from copy import copy
import json
from typing import List
from datetime import datetime
from konlpy.tag import Okt
from bertopic import BERTopic
import numpy as np
import re
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_vectors
from tqdm import tqdm
import os

# from backports.datetime_fromisoformat import MonkeyPatch # patching for python < 3.7
# MonkeyPatch.patch_fromisoformat()
okt = Okt()

TIME_GAP_INTO_EVENT = True

# imported from https://gist.github.com/sebleier/554280
stopwords_en = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
regex_en = re.compile(r'[^a-zA-Z|\s]+')
regex_ko = re.compile(r'[^ㄱ-ㅣ|가-힣|\s]+')
# %%

def extract_nouns(query: str):
  query_en = regex_en.sub('', query)
  query_ko = regex_ko.sub('', query)
  query_nouns = okt.nouns(query_ko) + query_en.split()

  return query_nouns

def isReformulated(query: str, prev_query: str):

  if prev_query == None:
    return False

  if edit_distance(query, prev_query) < 3:
    return True

  query_nouns = set(extract_nouns(query))
  prev_query_nouns = set(extract_nouns(prev_query))

  union = query_nouns.union(prev_query_nouns)
  intersect = query_nouns.intersection(prev_query_nouns)

  if len(union) <= 0:
    print(query, prev_query)
  return (len(intersect) / len(union) > 0.7) if len(union) > 0 else False

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
    return c
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

def edit_distance(s1, s2):
  if len(s1) < len(s2):
    return edit_distance(s2, s1)

  if len(s2) == 0:
    return len(s1)

  decomposed_s1 = [comp for c in s1 for comp in decompose(c)]
  decomposed_s2 = [comp for c in s2 for comp in decompose(c)]

  return levenshtein(decomposed_s1, decomposed_s2)


# %%

class Base:
  def __init__(self, sessionId: str, userId: str, time: str, Type: str):
    self.Type = Type
    self.sessionId = sessionId
    self.userId = userId
    self.time = datetime.fromisoformat(time)

  def reprJSON(self):
    return self.__dict__

class Session:
  def __init__(self, sessionId: str, userId: str, metrics: dict, sequence: List[Base], timestamp: datetime):
    self.sessionId = sessionId
    self.userId = userId
    self.metrics = metrics
    self.sequence = sequence
    self.BERTopicsKeywordCluster = -1
    self.ClusterID = -1
    self.timestamp = timestamp

  def reprJSON(self):
    temp = {
      "SessionNum": self.sessionId,
      "UserID": self.userId,
      "ClusterID": self.ClusterID,
      "BERTopicsKeywordCluster": self.BERTopicsKeywordCluster,
      "Sequence": self.sequence,
      "Timestamp": self.timestamp
    }

    # temp |= self.metrics
    temp.update(self.metrics)


    return temp

  def extract_queries(self):
    queries = [e for e in self.sequence if isinstance(e, Query)]
    return ' '.join([' '.join(extract_nouns(e.query)) for e in queries])

class Query(Base):
  def __init__(self, query: str, summary: dict, queryResults: List[dict], sessionId: str, userId: str, isRelated: bool, time: str, extendedQuery: str, previousQuery: str):
    self.query = query
    self.queryResults = queryResults
    self.summary = summary
    self.isRelated = isRelated
    self.extendedQuery = extendedQuery
    self.isRefined = isReformulated(query, previousQuery)
    self.isRepeated = (query == previousQuery)
    eventName = 'RefinedQuery' if self.isRefined else 'NewQuery'
    if TIME_GAP_INTO_EVENT:
      if self.summary['total_stay_sec'] < 30:
        eventName += '_Short'
    super().__init__(sessionId = sessionId, userId = userId, time = time, Type = eventName)

  def setIntermediateMetrics(self, metrics: dict):
    self.intermediateMetrics = metrics

  def reprJSON(self):
    temp = {
      "Query": self.query,
      "Type": self.Type,
      "QueryResults": self.queryResults,
      "Summary": self.summary,
      "IsRelated": self.isRelated,
      "ExtendedQuery": self.extendedQuery,
      "IsRefined": self.isRefined,
    }

    if hasattr(self, 'intermediateMetrics'):
      # temp.update(self.intermediateMetrics)

      temp |= self.intermediateMetrics

    return temp


class Click(Base):
  def __init__(self, data: dict, sessionId: str, userId: str, query: str):
    self.Query = query
    self.target_doc_url = data['argument']['target_doc_url']
    self.target_doc_seq = data['argument']['target_doc_seq']
    self.target_doc_title = data['argument']['target_doc_title']
    self.target_doc_context = data['argument']['target_doc_context']
    self.stay_time_sec = data['stay_time_sec']

    eventName = ''

    if self.target_doc_seq < 6:
      eventName = 'Click1-5'
    elif self.target_doc_seq < 11:
      eventName = 'Click6-10'
    else:
      eventName = 'Click11+'

    if TIME_GAP_INTO_EVENT:
      if self.stay_time_sec < 30:
        eventName += '_Short'

    super().__init__(sessionId = sessionId, userId = userId, time = data['timestamp'], Type = eventName)

  def reprJSON(self):
    return {
      "Query": self.Query,
      "Type": self.Type,
      "ClickedURL": self.target_doc_url,
      "Rank": self.target_doc_seq,
      "ClickedTitle": self.target_doc_title,
      "ClickedContext": self.target_doc_context,
      "DwellTime": self.stay_time_sec,
    }

class ClickQuickLink(Base):
  def __init__(self, data: dict, sessionId: str, userId: str, query: str):
    self.query = query
    self.req = data['argument']['req']
    self.match_keyword = data['argument']['match_keyword']
    self.source = data['argument']['source']
    self.title = data['argument']['title']

    super().__init__(sessionId = sessionId, userId = userId, time = data['timestamp'], Type='ClickQuickLink')

class EndSession(Base):

  def __init__(self, data: dict, sessionId: str, userId: str, query: str):
    self.query = query
    self.by = data['argument']['by']

    super().__init__(sessionId = sessionId, userId = userId, time = data['timestamp'], Type='EndSession')



class ComplexEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, Base):
      return obj.reprJSON()
    elif isinstance(obj, Session):
      return obj.reprJSON()
    elif isinstance(obj, datetime):
      return obj.isoformat()
    else:
      return json.JSONEncoder.default(self, obj)

def average_precision_score(ranks: List[int]):
  [((idx + 1) / rank) for idx, rank in enumerate(ranks)]

# %%

def compute_intermediate_metrics(queryEvent: Query, clickEvents: List[Base]):
  reciprocal_ranks = [(1 / c.target_doc_seq) for c in clickEvents if isinstance(c, Click)]

  ranks = [c.target_doc_seq for c in clickEvents if isinstance(c, Click)]

  meanRR = (sum(reciprocal_ranks) / len(reciprocal_ranks)) if len(reciprocal_ranks) > 0 else 0
  # maxRR = max(reciprocal_ranks) if len(reciprocal_ranks) > 0 else 0

  averagePrecision = sum([((idx + 1) / rank) for idx, rank in enumerate(ranks)]) / len(ranks) if len(ranks) > 0 else 0

  isAbandoned = queryEvent.summary['is_select_doc'] == False and queryEvent.summary['is_select_top5_doc'] == False

  isClickTop5 = queryEvent.summary['is_select_top5_doc']

  if len(ranks) > 0:
    dcg = np.sum([(1 / np.log2(r + 1)) for r in ranks])
    idcg = np.sum([(1 / np.log2((i + 1) + 1)) for i in range(len(ranks))]) 
    ndcg = dcg / idcg # Naive NDCG, without any consideration on clicks

  else:
    ndcg = 0

  # pSkip = 1 - (len(set(ranks)) / max(ranks)) if len(ranks) > 0 else 0

  return {
    'meanRR': meanRR,
    # 'maxRR': maxRR,
    'isAbandoned': isAbandoned,
    'isClickTop5': isClickTop5,
    'NDCG': ndcg,
    'AveragePrecision': averagePrecision,
    # 'pSkip': pSkip
  }

def compute_stats(session: List[Base]):
  queryEvents = [e for e in session if isinstance(e, Query)]
  totalQueryCount = len(queryEvents)

  meanRR = sum([q.intermediateMetrics['meanRR'] for q in queryEvents if q.intermediateMetrics['meanRR'] > 0]) / len([q.intermediateMetrics['meanRR'] for q in queryEvents if q.intermediateMetrics['meanRR'] > 0]) if len([q.intermediateMetrics['meanRR'] for q in queryEvents if q.intermediateMetrics['meanRR'] > 0]) > 0 else 0

  # maxRR = sum([q.intermediateMetrics['maxRR'] for q in queryEvents if q.intermediateMetrics['maxRR'] > 0]) / len([q.intermediateMetrics['maxRR'] for q in queryEvents if q.intermediateMetrics['maxRR'] > 0])
  
  abandonmentRate = len([q for q in queryEvents if q.intermediateMetrics['isAbandoned']]) / totalQueryCount
  reformulationRate = len([q for q in queryEvents if q.isRefined]) / totalQueryCount
  clickTop5Rate = len([q for q in queryEvents if q.intermediateMetrics['isClickTop5']]) / totalQueryCount
  ndcg = sum([q.intermediateMetrics['NDCG'] for q in queryEvents]) / len([q.intermediateMetrics['NDCG'] for q in queryEvents])
  map = sum([q.intermediateMetrics['AveragePrecision'] for q in queryEvents]) / len([q.intermediateMetrics['AveragePrecision'] for q in queryEvents])
  # pSkip = sum([q.intermediateMetrics['pSkip'] for q in queryEvents]) / len([q.intermediateMetrics['pSkip'] for q in queryEvents])
  return {
    'MeanRR': meanRR,
    # 'MaxRR': maxRR,
    'AbandonmentRate': abandonmentRate,
    'ReformulationRate': reformulationRate,
    'Click@1-5': clickTop5Rate,
    'NDCG': ndcg,
    'MAP': map
    # 'pSkip': pSkip
  }


def assign_cluster_id(sessions: List[Session]):
  # topic_model = BERTopic(embedding_model = 'distiluse-base-multilingual-cased-v1')
  # topic_model = BERTopic(embedding_model = 'paraphrase-multilingual-mpnet-base-v2')
  wv = load_facebook_vectors('')
  topic_model = BERTopic(embedding_model=wv)
  queries = [s.extract_queries() for s in sessions]

  topics, prob = topic_model.fit_transform(queries)
  topics = np.asarray(topics)
  all_topics = topic_model.get_topics()

  for i in range(len(topics)):
    sessions[i].BERTopicsKeywordCluster = int(topics[i])

  with open('BERTopics-clusters.json', 'w') as f:
    json.dump(all_topics, f, ensure_ascii=False, indent = 2)



# %%

# %%
# sessions_total = []

# for txt in os.listdir('../../log_data'):

#   with open(f'../../log_data/{txt}', 'r') as f:


with open('./SDS-log-sample-new.json', 'r') as f:
  data = json.load(f)

hits = data['hits']['hits']

sessionIds = set([h['_source']['session_id'] for h in hits]) # extract unique session ids

sessions_total = []

for sessionId in tqdm(sessionIds):
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
      extendedQuery = entry['_source']['query']['extend_keyword'],
      previousQuery=entry['_source']['summary']['prev_query_keyword']
    )

    clickEvents = [process_action(a, sessionId, userId, query) for a in entry['_source']['actions']]
    if len(clickEvents) == 0 and queryEvent.isRepeated:
      continue
    events.append(queryEvent)

    events.extend(clickEvents)

    intermediateStats = compute_intermediate_metrics(queryEvent = queryEvent, clickEvents = clickEvents)
    queryEvent.setIntermediateMetrics(intermediateStats)

  sorted_events = sorted(events, key = lambda x: x.time)
  session_metrics = compute_stats(sorted_events)

  new_session = Session(
    sessionId = sessionId,
    userId=userId,
    metrics=session_metrics,
    sequence = sorted_events,
    timestamp = events[0].time
  )

  sessions_total.append(new_session)

  # store them into the keyword_cluster format

  # process events array to make sth

def duplicate_sessions(s: Session, n = 100):
  sessions = []
  for i in range(0, n):
    new_s = copy(s)
    s.sessionId = s.sessionId + '-' + str(i)
    sessions.append(new_s)

  return sessions



new_sessions = [duplicate_sessions(s) for s in sessions_total]
new_sessions = [item for sublist in new_sessions for item in sublist]

assign_cluster_id(new_sessions)

# %%
print(sessions_total[0].sequence)
# %%

sessions_total[0].sequence[0].__dict__

# %%

# string = json.dumps(sessions_total, cls=ComplexEncoder, ensure_ascii=False)
# print(string)
# print('done')

with open('keyword_cluster_sessions.json', 'w') as f:
  json.dump(new_sessions, f, cls=ComplexEncoder, ensure_ascii=False, indent = 2)
# %%
