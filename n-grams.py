# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import random
from scipy import spatial
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import spacy
from scipy.cluster.hierarchy import average, dendrogram
from scipy.stats import chisquare, chi2_contingency
from rhc import mutual_info
from rhc import recursiveHierarchicalClustering
import logging
from datetime import datetime
import json
from bertopic import BERTopic
from sentence_transformers import util, SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import CountVectorizer

np.seterr(all='raise')

nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer('all-mpnet-base-v2')

stopwords = nlp.Defaults.stop_words
MIN_CLUSTER_SIZE = 100
# %%

if __name__ == '__main__':
  myLogger = logging.getLogger("my")
  myLogger.setLevel(logging.INFO)

  file_handler = logging.FileHandler('n-gram.log')
  myLogger.addHandler(file_handler)

  myLogger.info(f'Start time: {datetime.now()}')

# %%
queries = pd.read_csv('new_logs.csv', index_col="idx", dtype={"AnonID": "Int64", "Query": "string", "QueryTime": "string", "ItemRank": "Int32", "ClickURL": "string", "Type": "string", "SessionNum": "Int32"}, keep_default_na=False, na_values=[""])

queries = queries.loc[queries['Query'] != '-']


# %%




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
        maxRank = max(cur_clicks)
        minRank = min(cur_clicks)
        max_RRs.append(1 / minRank)
        mean_RRs.append(sum([(1 / r) for r in cur_clicks]) / len(cur_clicks))
        viewedItems = len(set(cur_clicks))
        p_skips.append(1 - (viewedItems / maxRank))
      else:
        abandon_counts += 1

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
      cur_clicks.append(row["ItemRank"] if row["ItemRank"] > 0 else 1)
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
      elif row["ItemRank"] >= 6 and row["ItemRank"] < 10:
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
  abandonment_rate = abandon_counts / num_queries
  reformulation_rate = reformulate_counts / num_queries
  
  clicks_per_query = clicks / num_queries

  return flattened, (p_skip, click1, max_RR, mean_RR, abandonment_rate, reformulation_rate, num_queries, clicks_per_query), sequences


def merge_clusters(cluster_info_list, distinguishing_features, clusters, vectors): 

  for key, cluster in cluster_info_list.items():
    if cluster["type"] == "Merged":
      continue
    merge_one_level(cluster_info_list, cluster, key, distinguishing_features, clusters, vectors)

def count_features(cluster_id, distinguishing_features, clusters, vectors):

  distinguishing_feature = [f for idx, features in distinguishing_features[cluster_id] for f, score in features]
  selected_vector_idx = np.where(clusters == cluster_id)[0]
  # print(selected_vector_idx)
  selected_vector_size = len(selected_vector_idx)
  selected_vectors = vectors[selected_vector_idx].copy()
  # selected_vectors[:, distinguishing_features[cluster_id]] = 0
  selected_vectors[:, distinguishing_feature] = 0

  counts = np.sum(selected_vectors, axis=0)

  vector_idxs = np.argpartition(counts, -3)[-3:]

  features = [(int(i), int(counts[i])) for i in vector_idxs]

  return features


def merge_one_level(cluster_info_list, cluster_info, cluster_id, distinguishing_features, clusters, vectors):

  if cluster_info["children"] is None:
    return

  out_cluster_id = cluster_info["out_cluster_id"]
  out_cluster = cluster_info_list[out_cluster_id]

  if out_cluster["children"] is None:
    features = count_features(out_cluster_id, distinguishing_features, clusters, vectors)
    distinguishing_features[out_cluster_id] = distinguishing_features[cluster_id] + [(out_cluster_id, features)]
    out_cluster["type"] = 'Remainder'
  else:
    merge_one_level(cluster_info_list, out_cluster, out_cluster_id, distinguishing_features, clusters, vectors)
    cluster_info["children"] = [cluster_info["in_cluster_id"]] + out_cluster["children"]
    out_cluster["type"] = "Merged"






def create_n_gram(sequence, n):
  if n < 2:
    return []
  ngram = []
  length = len(sequence)
  if (length < n):
    if (length == n - 1):
      filled = sequence + ["Empty" for i in range(n - length)]
      ngram.append(tuple(filled))

  else:
    for i in range(length - n + 1):
      ngram.append(tuple(sequence[i:i+n]))

  return ngram + create_n_gram(sequence, n - 1)


# def compute_distance(n_gram_1, n_gram_2, concatenated_set):
#   counter_ngram_1 = Counter(n_gram_1)
#   counter_ngram_2 = Counter(n_gram_2)


#   n_gram_vec_1 = np.asarray([(counter_ngram_1[n]) for n in concatenated_set])
#   n_gram_vec_1 = n_gram_vec_1 / np.linalg.norm(n_gram_vec_1)
#   n_gram_vec_2 = np.asarray([(counter_ngram_2[n]) for n in concatenated_set])
#   n_gram_vec_2 = n_gram_vec_2 / np.linalg.norm(n_gram_vec_2)

#   distance = 2 * np.arccos((np.dot(n_gram_vec_1, n_gram_vec_2) / np.sqrt(np.dot(n_gram_vec_1, n_gram_vec_1) * np.dot(n_gram_vec_2, n_gram_vec_2)))) / np.pi

#   return distance

def ngrams_to_vectors(n_gram_1, concatenated_set):
  counter_ngram_1 = Counter(n_gram_1)
  n_gram_vec_1 = np.asarray([(counter_ngram_1[n]) for n in concatenated_set])
  # n_gram_vec_1 = n_gram_vec_1 / np.linalg.norm(n_gram_vec_1)

  return n_gram_vec_1


# def generate_sequences_and_n_grams(group_by_sessions, sample, n = 5):
#   sequences = []
#   ngrams = []

#   for s in sample:
#     g = group_by_sessions.get_group(s)
#     flattened_log = flatten_logs(g)
#     ngram = create_n_gram(flattened_log, n)
#     sequences.append(flattened_log)
#     ngrams.append(ngram)

#   concat_set = list(set([ngram for n in ngrams for ngram in n]))

#   return sequences, ngrams, concat_set

def generate_n_grams(sequences, n = 5):
  ngrams = [create_n_gram(seq, n) for seq in sequences]

  concat_set = list(set([ngram for n in ngrams for ngram in n]))

  return ngrams, concat_set

def modularity(clusters, distances, m):
  num_clusters = len(clusters)
  # print(clusters)

  e_matrix = np.zeros((num_clusters, num_clusters))
  for i in range(len(clusters)):
    for j in range(i+1):
      selected_matrix = 1 - distances[np.ix_(clusters[i], clusters[j])]
      e_matrix[i, j] = np.sum(selected_matrix) / (m)
      if i == j:
        e_matrix[i, j] = e_matrix[i, j] / 2
        e_matrix[i, j] = (np.trace(selected_matrix) / (2 * m)) + e_matrix[i,j]
      e_matrix[j, i] = e_matrix[i, j]

  modularity = np.trace(e_matrix) - np.sum(np.dot(e_matrix, e_matrix))

  # print(e_matrix)

  return modularity

# def normalize_vectors(vectors):
#   # average_vector = np.average(vectors, axis=0)
#   normalized_vectors = vectors #  - average_vector

#   normalized_vectors = normalize(normalized_vectors, axis = 1)

#   return normalized_vectors

def compute_polar_distance(n_gram_vec_1, n_gram_vec_2):
  distance = 2 * np.arccos((np.dot(n_gram_vec_1, n_gram_vec_2) / np.sqrt(np.dot(n_gram_vec_1, n_gram_vec_1) * np.dot(n_gram_vec_2, n_gram_vec_2)))) / np.pi
  return distance

def getHalfPoint(scores):
	values = [x[1] for x in scores]
	total = sum(values) * 3 / 4
	cur = 0
	idx = 0
	for (idx, score, featureNum) in scores:
		cur += score
		if cur > total:
			break
	# print('half point stats', idx, min(values), max(values))
	return idx

def getSweetSpotL(evalResults):
	if len(evalResults) == 0: return 0
	cutoff = currentKnee = evalResults[-1][0]
	# print(evalResults)
	lastKnee = currentKnee + 1

	while currentKnee < lastKnee:
		lastKnee = currentKnee
		currentKnee = LMethod([item for item in evalResults if item[0] <= cutoff])
		# print(currentKnee)
		cutoff = currentKnee * 2

	return currentKnee

# tool used for finding out the defining feature threshold
# performs linear regression
def linearReg(evalResults):
	x = np.array([xv for (xv,yv, featureNum) in evalResults])
	y = np.array([yv for (xv,yv, featureNum) in evalResults])
	# print(x, y)
	A = np.vstack([x, np.ones(len(x))]).T
	result = np.linalg.lstsq(A,y)
	m, c = result[0]
	residual = result[1]
	# print(result)
	return ((m, c), residual if len(evalResults) > 2 else 0)

# using the L-Method, find of the sweetspot for getting defining features
def LMethod(evalResults):
	# print(len(evalResults))
	if len(evalResults) < 4:
		return len(evalResults)
	# the partition point c goes from the second point to the -3'th point
	minResidual = np.inf
	minCons = None
	minCutoff = None
	for cidx in range(1, len(evalResults) - 2):
		(con1, residual1) = linearReg(evalResults[:cidx + 1])
		(con2, residual2) = linearReg(evalResults[cidx + 1:])
		if (residual1 + residual2) < minResidual:
			minCons = (con1, con2)
			minCutoff = cidx
			minResidual = residual1 + residual2
	if minCutoff == None:
		print(('minCutoff is None', evalResults))
	return evalResults[minCutoff][0]

def divisive_clustering(ngrams, concat_set, n_clusters, k):
  vectors = np.asarray([ngrams_to_vectors(ngram, concatenated_set=concat_set) for ngram in ngrams])

  clusters = np.asarray([1 for vector in vectors])

  distinguishing_features = {1: []}

  len_vectors = len(vectors)

  whole_distances = np.zeros((len_vectors, len_vectors))

  for i in range(len_vectors):
    for j in range(i + 1):
      whole_distances[i, j] = compute_polar_distance(vectors[i], vectors[j]) #compute_distance(ngrams[i], ngrams[j], concatenated_set=concat_set)
      whole_distances[j, i] = whole_distances[i, j]

    clusters_info = {
      1: {
        "type": 'Tree',
        "cluster_size": len(vectors),
        "diameter": np.max(whole_distances),
        "children": None
      }
    }
  m = np.sum((1 - whole_distances))

  for i in range(n_clusters):
    myLogger.info('%dth division' % (i))

    idx = -1
    max_cluster_size = 0
    max_diameter = 0
    for key, cluster in clusters_info.items():
      # if cluster["diameter"] < 0.05:
      #   cluster["type"] = 'Leaf'
      if cluster["cluster_size"] < MIN_CLUSTER_SIZE:
        cluster["type"] = 'Leaf'
      if cluster["cluster_size"] > max_cluster_size and cluster["type"] == 'Tree' and cluster["children"] is None:
        idx = key    
        max_cluster_size = cluster["cluster_size"]
        max_diameter = cluster["diameter"]

    if idx == -1:
      break
    
    divide(vectors, clusters, distinguishing_features, clusters_info, idx, whole_distances, k)
    score = silhouette_score(whole_distances, clusters, metric='precomputed')

    mod_clusters = [np.where(clusters == i)[0] for i in np.unique(clusters)]
    modularity_score = modularity(mod_clusters, whole_distances, m)

    myLogger.info("Modularity score at iteration %d: %f" % (i, modularity_score))
    myLogger.info("Silhouette_score at iteration %d: %f" % (i, score))

  return clusters, distinguishing_features, vectors, clusters_info

def divide(vectors, clusters, distinguishing_features, clusters_info, cluster_id, whole_distances, k):

  myLogger.info('Dividing cluster %d' % cluster_id)

  distinguishing_feature = [f for idx, features in distinguishing_features[cluster_id] for f, score in features]
  selected_vector_idx = np.where(clusters == cluster_id)[0]
  # print(selected_vector_idx)
  selected_vector_size = len(selected_vector_idx)
  selected_vectors = vectors[selected_vector_idx].copy()
  # selected_vectors[:, distinguishing_features[cluster_id]] = 0
  selected_vectors[:, distinguishing_feature] = 0

  # print(distinguishing_features[cluster_id])

  distances = np.zeros((selected_vector_size, selected_vector_size))

  if cluster_id == 1:
    distances = whole_distances
  else:
    for i in range(selected_vector_size):
      for j in range(i+1):
        try:
          if np.any(selected_vectors[i]) == False or np.any(selected_vectors[j]) == False:
            distances[i, j] = 0
            distances[j, i] = 0
            continue
          distances[i, j] = compute_polar_distance(selected_vectors[i], selected_vectors[j]) #compute_distance(ngrams[i], ngrams[j], concatenated_set=concat_set)
          distances[j, i] = distances[i, j]
        except:
          print(sequences[i], sequences[j])
          print(vectors[i], vectors[j])
          print(selected_vectors[i], selected_vectors[j])
          # print([concat_set_dict[n][idx] for idx in distinguishing_features[cluster_id]])
          print([concat_set_dict[n][idx] for idx in distinguishing_feature])



  # splinters = [np.argmax(np.sum(distances, axis = 0))]

  mask = np.ones(selected_vector_size, bool)

  mask[np.argmax(np.average(distances, axis = 1))] = False
  # print(np.argmax(np.average(distances, axis = 1)))
  while True:
    avg_distance_to_splinters = np.average(distances[:, ~mask], axis=1)

    avg_distance_within = np.average(distances[:, mask], axis=1)

    diff = avg_distance_to_splinters - avg_distance_within

    diff[~mask] = np.Inf
    candidate = np.argmin(diff)
    # print(candidate, diff[candidate], mask[candidate])

    if diff[candidate] < 0:
      mask[candidate] = False
    else:
      break

  # splinter_cluster = selected_vectors[~mask]
  # remaining_cluster = selected_vectors[mask]

  in_clusters = selected_vector_idx[~mask]
  out_clusters = selected_vector_idx[mask]

  if (in_clusters.size < MIN_CLUSTER_SIZE * 0.5) or (out_clusters.size < MIN_CLUSTER_SIZE * 0.5):
    clusters_info[cluster_id]['type'] = 'Leaf'
    return

  in_cluster_id = max(clusters_info.keys()) + 1
  out_cluster_id = max(clusters_info.keys()) + 2

  clusters_info[cluster_id]["in_cluster_id"] = in_cluster_id
  clusters_info[cluster_id]["out_cluster_id"] = out_cluster_id

  clusters_info[cluster_id]["children"] = [in_cluster_id, out_cluster_id]
  
  # clusters_info[cluster_id]["children"].append(in_cluster_id)

  clusters[in_clusters] = in_cluster_id
  clusters[out_clusters] = out_cluster_id

  # contingency_table = np.asarray([np.sum(splinter_cluster, axis=0), np.sum(remaining_cluster, axis=0)])

  max_chisq = 0
  max_idx = 0
  chisqs = []
  for i in range(vectors.shape[1]):
    if i in distinguishing_feature:
      continue
    splinter_cluster_dist = vectors[in_clusters, i]
    remaining_cluster_dist = vectors[out_clusters, i]
    chi2 = mutual_info.chi_square(remaining_cluster_dist, 0, splinter_cluster_dist, 0)
    # cont = np.asarray([[np.where(splinter_cluster[:, i] > 0)[0].size, np.where(splinter_cluster[:, i] <= 0)[0].size],
    # [np.where(remaining_cluster[:, i] > 0)[0].size, np.where(remaining_cluster[:, i] <= 0)[0].size]])
    # if not np.all(cont):
    #   continue
    # # print(cont)
    # chi2, p, dof, ex = chi2_contingency(cont)
    # print("Chi sq for cluster %d, component %d: %d" % (cluster_id, i, chi2))
    chisqs.append((i, chi2))
    if chi2 > max_chisq:
      max_chisq = chi2
      max_idx = i
  chisqs = sorted(chisqs, key= lambda x: x[1], reverse = True)
  chisqs = [(idx, score[1], score[0]) for idx, score in enumerate(chisqs)]
  halfpoint = getHalfPoint(chisqs)
  res = getSweetSpotL(chisqs[:max(200, 2 * halfpoint)])
  myLogger.info('Initial res: %d' % res)
  if res > k:
    print(res)
    res = k
  cutoff_features = [(x[2], x[1]) for x in chisqs[:(res+1)]]
  # print(chisqs, res, cutoff_features)
  # cutoff_point = next(i for i,v in enumerate(chisqs) if v[0] == res)
  # cutoff_features = [x[0] for x in chisqs[:cutoff_point]]
  # print(chisqs)
  # print(res)
  # print(cutoff_features)

  distinguishing_features[in_cluster_id] = distinguishing_features[cluster_id] + [(in_cluster_id, cutoff_features)]
  distinguishing_features[out_cluster_id] = distinguishing_features[cluster_id] # + cutoff_features

  in_cluster_dist = whole_distances[np.ix_(in_clusters, in_clusters)]
  out_cluster_dist = whole_distances[np.ix_(out_clusters, out_clusters)]


  clusters_info[in_cluster_id] = {
    "type": 'Tree',
    "cluster_size": in_clusters.shape[0],
    "diameter": np.max(in_cluster_dist),
    "children": None
  }


  clusters_info[out_cluster_id] = {
    "type": 'Tree',
    "cluster_size": out_clusters.shape[0],
    "diameter": np.max(out_cluster_dist),
    "children": None
  }

# %%



# %%
group_by_sessions = queries.groupby('AnonID', sort=False, as_index=False)
qqq = group_by_sessions.max(numeric_only=True)
newIDs = qqq.loc[qqq['SessionNum'] >= 19]

# %%
group_by_sessions = queries.groupby(["AnonID", "SessionNum"])

group_by_sessions = group_by_sessions.filter(lambda x: not ((
  x["Query"].str.contains('porn|fuck|nude|sex|www|\.com|\.net', regex=True).any()
)))

group_by_sessions = group_by_sessions.groupby(["AnonID", "SessionNum"])


tuples = []
# for idx, row in newIDs.iterrows():
#   tuples += [(row["AnonID"], i) for i in range(row["SessionNum"])]

## First, extract all tuples with appropriately long sessions

# ssss = group_by_sessions.count()
# ss = ssss # [ssss["Query"] >= 5]

ssss = group_by_sessions.nunique()
ss = ssss[ssss["Query"] <= 5]

for idx, _ in ss.iterrows():
  tuples += [idx]

# %%
### Then, draw 5,000 samples

SAMPLE_SIZE = 10000

random.seed(3)

sample = random.sample(tuples, SAMPLE_SIZE)

sequences = []
json_seqs = []
query_text = []

for s in sample:
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
    "BERTopicsKeywordCluster": 0,
    "KMeansCluster": 0
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
        print(i)
        top_n_words[label] = [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1]
#     top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

cc = KMeans(n_clusters=N_CLUSTERS)

query_embeddings = model.encode(query_text, convert_to_numpy=True)
query_embeddings = query_embeddings.astype('float64')
kmeans_labels = cc.fit_predict(query_embeddings)

docs = []

for i in range(N_CLUSTERS):
    idxs = np.argwhere(kmeans_labels == i)
    doc = ''
    for j in idxs:
        doc += (query_text[j[0]] + ' ')
    docs.append(doc)
  
tf_idf, count = c_tf_idf(docs, m=SAMPLE_SIZE)
top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs)

# %% BERTopic Clustering parts

topic_model = BERTopic(embedding_model = model)
topics, prob = topic_model.fit_transform(query_text)

all_topics = topic_model.get_topics()

# %% Output 

for i in range(len(topics)):
  json_seqs[i]['BERTopicsKeywordCluster'] = int(topics[i])
  json_seqs[i]['KMeansCluster'] = int(kmeans_labels[i])

with open('BERTopics-cluster.json', 'w') as f:
  json.dump(all_topics, f, ensure_ascii=True, indent = 2)

with open('KMeans-clusters.json', 'w') as f:
  json.dump(top_n_words, f, ensure_ascii=True, indent=2)

# %%
seq_dict = {}
ngram_dict = {}
concat_set_dict = {}
clusters_dict = {}
distinguishing_features_dict = {}
vectors_dict = {}
clusters_info_dict = {}


# %%

for k in [20]:
  for n in range(5, 6):
    
    ngram_dict[n], concat_set_dict[n] = generate_n_grams(sequences, n = n)
    clusters_dict[n], distinguishing_features_dict[n], vectors_dict[n], clusters_info_dict[n] = divisive_clustering(ngram_dict[n], concat_set_dict[n], 100, k)
    score = silhouette_score(vectors_dict[n], clusters_dict[n], metric='cosine')
    myLogger.info("n-gram size: %f, silhouette score: %f" % (n, score))

    merge_clusters(clusters_info_dict[n], distinguishing_features_dict[n], clusters_dict[n], vectors_dict[n])

    with open(f'n-gram-{n}-{k}.txt', 'w') as f:
      f.write(','.join(['clusterId', 'sequences', 'userId', 'sessionNum', 'initialQuery']))
      f.write('\n')
      for i in range(SAMPLE_SIZE):
        label_divisive = str(clusters_dict[n][i])
        userid = str(sample[i][0])
        group = str(sample[i][1])
        g = group_by_sessions.get_group(sample[i])
        previous_query = g.iloc[0]['Query']
        row = label_divisive + ',' + '+'.join(sequences[i]) + ',' + userid + ',' + group + ',' + previous_query + '\n'
        f.write(row)
    
    # with open(f'cluster-info-{n}-{k}.txt', 'w') as f:
    #   for key, cluster_info in clusters_info_dict[n].items():
    #     f.write("Cluster %d: diameter %f, size %d, type %s \n" % (key, cluster_info["diameter"], cluster_info["cluster_size"], cluster_info["type"]))
    #     children = str(cluster_info["children"])
    #     f.write(f"Children {children} \n")
    #     f.write("Distinguishing factors\n")
    #     f.write(str([concat_set_dict[n][idx] for idx in distinguishing_features_dict[n][key]]))
    #     f.write("\n")

    with open(f'cluster-info-{n}-{k}.json', 'w') as f:
      tree = {
        "root_id": 1,
        "nodes": []
      }
      for key, cluster_info in clusters_info_dict[n].items():
        distinguishing_feature = [(cluster_id, f, score) for cluster_id, features in distinguishing_features_dict[n][key] for f, score in features]
        node = {
          "id": key,
          "label": cluster_info['type'],
          "distinguishing_features": [{
            "cluster_id": cluster_id,
            "action_items": concat_set_dict[n][idx],
            "score": score
          } for cluster_id, idx, score in distinguishing_feature],
          "divided_cluster": cluster_info["in_cluster_id"] if cluster_info["children"] is not None else None,
          "remaining_cluster": cluster_info["out_cluster_id"] if cluster_info["children"] is not None else None,
          "children": cluster_info["children"],
          "subtree_size": cluster_info["cluster_size"]
        }
        tree["nodes"].append(node)

      json.dump(tree, f, ensure_ascii=True, indent = 2)

    with open(f'sequences-{n}-{k}.json', 'w') as f:
      for i in range(SAMPLE_SIZE):
        label_divisive = clusters_dict[n][i]
        json_seqs[i]["ClusterID"] = int(label_divisive)
      json.dump(json_seqs, f, ensure_ascii=True, indent = 2)

# %%

  
  


# %%
## Trash, scratch cell

# with open('seq.txt', 'w') as f:
#   for idx, sequence in enumerate(sequences):
#     one_letter = {
#       'NewQuery': 'N',
#       'RefinedQuery': 'R',
#       'Click1': 'C',
#       'Click2-5': 'D',
#       'Click6-10': 'E',
#       'Click11+': 'F',
#       'NextPage': 'P'
#     }

#     seq = ''.join([(one_letter[action] + '(1)') for action in sequence])

#     f.write('%d\t%s\n' % (idx + 1, seq))

# with open('n-gram-test-divisive.txt', 'w') as f:
#   for i in range(SAMPLE_SIZE):
#     label_divisive = str(clusters_dict[3][i])
#     userid = str(sample[i][0])
#     group = str(sample[i][1])
#     g = group_by_sessions.get_group(sample[i])
#     previous_query = g.iloc[0]['Query']
#     row = label_divisive + ',' + '+'.join(sequences[i]) + ',' + userid + ',' + group + ',' + previous_query + '\n'
#     f.write(row)

# print([concat_set[i] for i in distinguishing_features])  

# print(concat_set[215])
# print(concat_set[232])

# print(clusters)
# print(distinguishing_features)
# cc = np.asarray(concat_set)
# print(cc[distinguishing_features[30]])

# print(vectors_dict[3][596])

# print(concat_set_dict[3][0])

# for key, cluster_info in clusters_info_dict[3].items():
#   print("Cluster %d: diameter %f, size %d" % (key, cluster_info["diameter"], cluster_info["cluster_size"]))
#   print("Children", cluster_info["children"])
#   print("Distinguishing factors")
#   print([concat_set_dict[3][idx] for idx in distinguishing_features_dict[3][key]])  

# print(distinguishing_features_dict[4])


