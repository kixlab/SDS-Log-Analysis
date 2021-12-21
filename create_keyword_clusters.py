# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import random
import spacy
import logging
from datetime import datetime
import json
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

np.seterr(all='raise')

nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer('all-mpnet-base-v2')

stopwords = nlp.Defaults.stop_words
MIN_CLUSTER_SIZE = 30
# %%

if __name__ == '__main__':
  myLogger = logging.getLogger("my")
  myLogger.setLevel(logging.INFO)

  file_handler = logging.FileHandler('n-gram.log')
  myLogger.addHandler(file_handler)

  myLogger.info(f'Start time: {datetime.now()}')

# %%
# queries = pd.read_csv('new_logs_new_nextpage.csv', index_col="idx", dtype={"AnonID": "Int64", "Query": "string", "QueryTime": "string", "ItemRank": "Int32", "ClickURL": "string", "Type": "string", "SessionNum": "Int32"}, keep_default_na=False, na_values=[""])

# mask = (queries['Query'] != '-') & (queries['Query'].str.len() >= 2)
# queries = queries[mask]
# queries = queries.loc[queries['Query'] != '-']
# queries = queries.loc[queries['Query'].str.len() >= 2]

# %%

def compute_edit_distance(queryA, queryB): # Character level
  a = len(queryA)
  b = len(queryB)
  dp = [[0 for x in range(b + 1)] for x in range(a + 1)]

  for i in range(a + 1):
    for j in range(b + 1):
      if i == 0:
        dp[i][j] = j
      elif j == 0:
        dp[i][j] = i
      elif queryA[i - 1] == queryB[j - 1]:
        dp[i][j] = dp[i-1][j-1]
      else:
        dp[i][j] = 1 + min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j])

  return dp[a][b]

def compute_shared_words(queryA, queryB):
  setA = set(queryA)
  setB = set(queryB)

  union = setA.union(setB)
  intersect = setA.intersection(setB)
  try:
    return len(intersect) / len(union)
  except:
    print(queryA, queryB)
    return 0

def compute_semantic_similarity(old_query, new_query):
  # embeddings = bc.encode([old_query, new_query])
  # return 1 - spatial.distance.cosine(embeddings[0], embeddings[1])

  return 1

def is_new_query(old_query, old_processed_query, new_query, new_processed_query):
  # compute semantic similarities and edit distances to gauge query similarities
  if compute_edit_distance(old_query, new_query) < 3: #
    return False

  if compute_shared_words(old_processed_query, new_processed_query) > 0.7:
    return False

  # if old_processed_query.similarity(new_processed_query) > 0.8:
  #   return False
    
  return True

def process_query(query):
  doc = nlp(query)
  tokens_wo_sw = [word.lemma_ for word in doc if ((word.is_stop == False) and (word.is_punct == False) and word.lemma_ != '-PRON-')]
  return tokens_wo_sw

def flatten_logs(logs_dataframe):
  previous_row = None
  previous_query = ''
  previous_processed_query = None
  flattened = []

  clicks1 = 0
  clicks = 0
  clicks1_flag = False
  cur_clicks = []
  max_RRs = []
  mean_RRs = []
  p_skips = []
  ndcgs = []
  num_queries = 0
  abandon_counts = 0
  reformulate_counts = 0

  sequences = []

  for idx, row in logs_dataframe.iterrows():
    if previous_row is None: # Every logstream starts with a query
      # print(row)
      flattened.append("NewQuery")
      previous_row = row
      previous_query = row['Query']
      previous_processed_query = process_query(previous_query)
      num_queries += 1

      sequences.append({
        "Type": "NewQuery",
        "Query": row['Query']
      })
      
      continue

    if row["Type"] == "Query":
      clicks1_flag = False
      if len(cur_clicks) > 0:
        dcg = np.sum([(1 / np.log2(r + 1)) for r in cur_clicks])
        idcg = np.sum([(1 / np.log2((i + 1) + 1)) for i in range(len(cur_clicks))])
        ndcg = dcg / idcg # Naive NDCG, without any consideration on clicks
        ndcgs.append(ndcg)
        maxRank = max(cur_clicks)
        minRank = min(cur_clicks)
        max_RRs.append(1 / minRank)
        mean_RRs.append(sum([(1 / r) for r in cur_clicks]) / len(cur_clicks))
        viewedItems = len(set(cur_clicks))
        p_skips.append(1 - (viewedItems / maxRank))
      else:
        abandon_counts += 1

      if not ((previous_query == row['Query']) and (sequences[-1]['Type'] == "NewQuery" or sequences[-1]['Type'] == "RefinedQuery")):
        new_processed_query = process_query(row['Query'])
        if is_new_query(previous_query, previous_processed_query, row['Query'], new_processed_query):
          flattened.append("NewQuery")
          previous_query = row['Query']
          previous_processed_query = new_processed_query
          sequences.append({
            "Type": "NewQuery",
            "Query": row['Query']
          })
        else:
          flattened.append("RefinedQuery")
          sequences.append({
            "Type": "RefinedQuery",
            "Query": row['Query']
          })
          reformulate_counts += 1

      cur_clicks = []
      num_queries += 1
    elif row["Type"] == "Click":
      clicks += 1
      cur_clicks.append(row["ItemRank"] if row["ItemRank"] > 0 else 1) # handling weird cases with rank  = 0
      if row["ItemRank"] < 6:
        flattened.append("Click1-5")
        if clicks1_flag == False:
          clicks1 +=1
          clicks1_flag = True
        sequences.append({
          "Type": "Click1-5",
          "Query": row['Query'],
          "Rank": row['ItemRank'],
          "ClickedURL": row['ClickURL']
        })
      # else:
      #   flattened.append("Click6+")
      # if row["ItemRank"] == 1:
      #   flattened.append("Click1")
      # elif row["ItemRank"] >= 2 and row["ItemRank"] < 6:
      #   flattened.append("Click2-5")
      elif row["ItemRank"] >= 6 and row["ItemRank"] < 11:
        flattened.append("Click6-10")
        sequences.append({
          "Type": "Click6-10",
          "Query": row['Query'],
          "Rank": row['ItemRank'],
          "ClickedURL": row['ClickURL']
        })
      else:
        flattened.append("Click11+")
        sequences.append({
          "Type": "Click11+",
          "Query": row['Query'],
          "Rank": row['ItemRank'],
          "ClickedURL": row['ClickURL']
        })
    elif row["Type"] == "NextPage":
      sequences.append({
        "Type": "NextPage",
        "Query": row['Query']
      })
      flattened.append("NextPage")

  if len(cur_clicks) > 0:
    maxRank = max(cur_clicks)
    minRank = min(cur_clicks)
    max_RRs.append(1 / minRank)
    mean_RRs.append(sum([(1 / r) for r in cur_clicks]) / len(cur_clicks))
    viewedItems = len(set(cur_clicks))
    p_skips.append(1 - (viewedItems / maxRank))
  else:
    if flattened[-1] == "NewQuery" or flattened[-1] == "RefinedQuery":
      abandon_counts += 1

  p_skip = sum(p_skips)/len(p_skips) if len(p_skips) > 0 else 1
  click1 = clicks1 / num_queries
  max_RR = sum(max_RRs) / len(max_RRs) if len(max_RRs) > 0 else 0
  mean_RR = sum(mean_RRs) / len(mean_RRs) if len(mean_RRs) > 0 else 0
  mean_NDCG = sum(ndcgs) / len(ndcgs) if len(ndcgs) > 0 else 0
  abandonment_rate = abandon_counts / num_queries
  reformulation_rate = reformulate_counts / num_queries
  
  clicks_per_query = clicks / num_queries

  return flattened, (p_skip, click1, max_RR, mean_RR, abandonment_rate, reformulation_rate, num_queries, clicks_per_query, mean_NDCG), sequences

# %%
# group_by_sessions = queries.groupby('AnonID', sort=False, as_index=False)

# # %%
# group_by_sessions = queries.groupby(["AnonID", "SessionNum"])

# group_by_sessions = group_by_sessions.filter(lambda x: not ((
#   x["Query"].str.contains('porn|fuck|nude|sex|www|\.com|\.net|shemale', regex=True).any()
# )))
# group_by_sessions.to_csv('filtered.csv', index_label="idx")

group_by_sessions = pd.read_csv('filtered.csv', index_col="idx", dtype={"AnonID": "Int64", "Query": "string", "QueryTime": "string", "ItemRank": "Int32", "ClickURL": "string", "Type": "string", "SessionNum": "Int32"}, keep_default_na=False, na_values=[""])
group_by_sessions = group_by_sessions.groupby(["AnonID", "SessionNum"])


tuples = []

## First, extract all tuples with appropriately long sessions

# ssss = group_by_sessions.count()
# ss = ssss # [ssss["Query"] >= 5]

ssss = group_by_sessions.nunique()
ss = ssss[ssss["Query"] <= 5]

for idx, _ in ss.iterrows():
  tuples += [idx]

# %%
### Then, draw 5,000 samples

SAMPLE_SIZE = 50000

random.seed(3)

sample = random.sample(tuples, SAMPLE_SIZE)

# sample = tuples

sequences = []
json_seqs = []
query_text = []

for s in tqdm(sample):
  g = group_by_sessions.get_group(s)
  flattened_log, value, json_seq = flatten_logs(g)
  queries = g[g['Type'] == "Query"]['Query']
  v = []
  queries_concat = queries.str.cat(sep='. ')
  query_text.append(queries_concat)
  sequences.append(flattened_log)
  json_seqs.append({
    "SessionNum": s[1],
    "UserID": s[0],
    "ClusterID": 0,
    "Sequence": json_seq,
    "pSkip": value[0],
    "Click@1-5": value[1],
    "MaxRR": value[2],
    "MeanRR": value[3],
    "AbandonmentRate": value[4],
    "ReformulationRate": value[5],
    "QueriesCount": value[6],
    "ClicksPerQuery": value[7],
    "NDCG": float(value[8]),
    "BERTopicsKeywordCluster": 0,
    "KMeansCluster": 0,
    "FinalBehavior": json_seq[-1]["Type"]
  })

# %% KMeans Topic Clustering parts
N_CLUSTERS = 100

def c_tf_idf(documents, m, ngram_range=(1, 3)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count

def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    labels = list(range(N_CLUSTERS))
    top_n_words = {}
    for i, label in enumerate(labels):
        # print(i)
        top_n_words[label] = [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1]
#     top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

# cc = KMeans(n_clusters=N_CLUSTERS)

# query_embeddings = model.encode(query_text, convert_to_numpy=True)
# query_embeddings = query_embeddings.astype('float64')
# kmeans_labels = cc.fit_predict(query_embeddings)

# docs = []

# for i in range(N_CLUSTERS):
#     idxs = np.argwhere(kmeans_labels == i)
#     doc = ''
#     for j in idxs:
#         doc += (query_text[j[0]] + ' ')
#     docs.append(doc)
  
# tf_idf, count = c_tf_idf(docs, m=SAMPLE_SIZE)
# top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs)

# %% BERTopic Clustering parts
topic_model = BERTopic.load('BERTopic_model_50000_new', embedding_model='all-mpnet-base-v2')

# for q in tqdm(range(int(len(query_text) / 10))):
#   try:
#     t, p = topic_model.transform(query_text[q*10:(q+1)*10])
#   except:
#     print(query_text[q*10:(q+1)*10])

topic_model = BERTopic(embedding_model = 'all-mpnet-base-v2', nr_topics="auto", min_topic_size=50)
topics, prob = topic_model.fit_transform(query_text)
topics = np.asarray(topics)
all_topics = topic_model.get_topics()

# # %% Output 

for i in range(len(topics)):
  json_seqs[i]['BERTopicsKeywordCluster'] = int(topics[i])
#  json_seqs[i]['KMeansCluster'] = int(kmeans_labels[i])

with open('BERTopics-cluster-50000-50-new.json', 'w') as f:
  json.dump(all_topics, f, ensure_ascii=True, indent = 2)

# # with open('KMeans-clusters.json', 'w') as f:
# #   json.dump(top_n_words, f, ensure_ascii=True, indent=2)

# # %%

with open(f'sequences-50000-new.json', 'w') as f:
  json.dump(json_seqs, f, ensure_ascii=True, indent = 2)

# topic_model.save("BERTopic_model_50000_new", save_embedding_model=False)




# %%
## Test cell

test_model = BERTopic.load('BERTopic_model_50000_new', embedding_model=model)
test_model.visualize_heatmap()
# test_model.visualize_topics()
# test_model.visualize_heatmap()

# %%
from sklearn.metrics.pairwise import cosine_similarity

topics = sorted(list(test_model.get_topics().keys()))
print(topics)

#%%
selected_topics = [-1, 1, 2, 3, 4, 5, 6, 7, 9, 14, 16, 17, 19, 20, 22, 24, 25, 27, 28, 31, 33, 35, 36, 38, 39, 40, 41, 42, 43, 45, 47, 48, 49, 50, 52, 53, 55, 56, 58, 59, 61, 65, 68, 69, 74, 76, 79]
embeddings = np.array(test_model.topic_embeddings)
indices = np.array([topics.index(topic) for topic in topics])
embeddings = embeddings[indices]
distance_matrix = cosine_similarity(embeddings)



with open('keyword-similarity.json', 'w') as g:
  json.dump(distance_matrix.tolist(), g)
# cosine_similarity
# %%


  
# %%
