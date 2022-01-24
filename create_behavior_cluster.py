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
MIN_CLUSTER_SIZE = 20
# %%

if __name__ == '__main__':
  myLogger = logging.getLogger("my")
  myLogger.setLevel(logging.INFO)

  file_handler = logging.FileHandler('n-gram.log')
  myLogger.addHandler(file_handler)

  myLogger.info(f'Start time: {datetime.now()}')

# %%
# queries = pd.read_csv('new_logs.csv', index_col="idx", dtype={"AnonID": "Int64", "Query": "string", "QueryTime": "string", "ItemRank": "Int32", "ClickURL": "string", "Type": "string", "SessionNum": "Int32"}, keep_default_na=False, na_values=[""])

# queries = queries.loc[queries['Query'] != '-']


# %%

# Create a sequence of actions from each session

def flatten_logs(sequences_list):
  
  result =  [[a["Type"] for a in s["Sequence"]] for s in sequences_list]

  return result




# %%

# in case of too small clusters, we need to count the number of times each feature appears in the cluster and add the most popular ones to the list of distinguishing features

def count_features(cluster_id, distinguishing_features, clusters, vectors):

  distinguishing_feature = [f for idx, features in distinguishing_features[cluster_id] for f, score in features]
  selected_vector_idx = np.where(clusters == cluster_id)[0]
  # print(selected_vector_idx)
  selected_vector_size = len(selected_vector_idx)
  selected_vectors = vectors[selected_vector_idx].copy()
  # selected_vectors[:, distinguishing_features[cluster_id]] = 0
  selected_vectors[:, distinguishing_feature] = 0

  counts = np.sum(selected_vectors, axis=0)

  if len(counts) > 3: # choose the top 3 features
    vector_idxs = np.argpartition(counts, -3)[-3:]
  else:
    vector_idxs = counts

  features = [(int(i), int(counts[i])) for i in vector_idxs]

  return features

# merge the cluster with its outer children, which does not have its own distinguishing features
def merge_one_level(cluster_info_list, cluster_info, cluster_id, distinguishing_features, clusters, vectors):

  if cluster_info["children"] is None:
    return

  # select the outer children of the cluster
  out_cluster_id = cluster_info["out_cluster_id"]
  out_cluster = cluster_info_list[out_cluster_id]

  if out_cluster["children"] is None: # Remainder cluster, which does not have any distinguishing features
    features = count_features(out_cluster_id, distinguishing_features, clusters, vectors)
    distinguishing_features[out_cluster_id] = distinguishing_features[cluster_id] + [(out_cluster_id, features)]
    out_cluster["type"] = 'Remainder'
  else: # merge the cluster with its outer children
    merge_one_level(cluster_info_list, out_cluster, out_cluster_id, distinguishing_features, clusters, vectors)
    cluster_info["children"] = [cluster_info["in_cluster_id"]] + out_cluster["children"]
    out_cluster["type"] = "Merged"

# transform the binary tree of clusters into a list of clusters

def merge_clusters(cluster_info_list, distinguishing_features, clusters, vectors): 

  for key, cluster in cluster_info_list.items():
    if cluster["type"] == "Merged":
      continue
    merge_one_level(cluster_info_list, cluster, key, distinguishing_features, clusters, vectors)

  for key, cluster in cluster_info_list.items(): 
    if cluster["type"] == "Merged":
      continue

    if cluster['cluster_size'] < MIN_CLUSTER_SIZE: # when the cluster is too small, select the most popular features as distinguishing features
      features = count_features(key, distinguishing_features, clusters, vectors)
      distinguishing_features[key] = [(k, f) for k, f in distinguishing_features[key] if k != key] + [(key, features)]

    # For clusters with all scores zero, select the most popular features as distinguishing features
    if sum([feature[1] for k, f in distinguishing_features[key] for feature in f]) == 0:
      features = count_features(key, distinguishing_features, clusters, vectors)
      distinguishing_features[key] = [(k, f) for k, f in distinguishing_features[key] if k != key] + [(key, features)]

# create n-grams from the list of actions in the session
def create_n_gram(sequence, n):
  if n < 2:
    return []
  ngram = []
  length = len(sequence)
  if (length < n):
    if (length == n - 1):
      filled = sequence + ["Empty" for i in range(n - length)] # fill the session with "Empty" if the session is shorter than n
      ngram.append(tuple(filled))

  else:
    for i in range(length - n + 1):
      ngram.append(tuple(sequence[i:i+n]))

  return ngram + create_n_gram(sequence, n - 1)

# count the number of times each n-gram appears in the session and make it as a vector
def ngrams_to_vectors(n_gram_1, concatenated_set):
  counter_ngram_1 = Counter(n_gram_1)
  n_gram_vec_1 = np.asarray([(counter_ngram_1[n]) for n in concatenated_set])

  return n_gram_vec_1

# create the set of n-grams from all n-grams in all sessions
def generate_n_grams(sequences, n = 5):
  ngrams = [create_n_gram(seq, n) for seq in sequences]

  concat_set = list(set([ngram for n in ngrams for ngram in n]))

  return ngrams, concat_set

# compute the modularity scores of the clustering results
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

  return modularity

# computes the polar distance between two vectors
def compute_polar_distance(n_gram_vec_1, n_gram_vec_2):
  distance = 2 * np.arccos((np.dot(n_gram_vec_1, n_gram_vec_2) / np.sqrt(np.dot(n_gram_vec_1, n_gram_vec_1) * np.dot(n_gram_vec_2, n_gram_vec_2)))) / np.pi
  return distance

# Helper functions for extracting the distinguishing features from the clusters
# adopted from https://sandlab.cs.uchicago.edu/clickstream/index.html

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


# divisivie hierarchical clustering of the sessions, based on the similarity of the n-gram occurrence vectors
def divisive_clustering(ngrams, concat_set, n_clusters, k, target = []):
  vectors = np.asarray([ngrams_to_vectors(ngram, concatenated_set=concat_set) for idx, ngram in enumerate(ngrams) if idx in target])

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
      "children": None,
      "parent": None
    }
  }
  m = np.sum((1 - whole_distances))

  for i in range(n_clusters):
    myLogger.info('%dth division' % (i))

    idx = -1
    max_cluster_size = 0
    max_diameter = 0
    for key, cluster in clusters_info.items(): # find the cluster with the largest size
      # if cluster["diameter"] < 0.05:
      #   cluster["type"] = 'Leaf'
      if cluster["cluster_size"] < min(MIN_CLUSTER_SIZE, 0.1 * len_vectors): # if the cluster is too small, it is a leaf cluster and we can stop
        cluster["type"] = 'Leaf'
      if cluster["cluster_size"] > max_cluster_size and cluster["type"] == 'Tree' and cluster["children"] is None:
        idx = key    
        max_cluster_size = cluster["cluster_size"]
        max_diameter = cluster["diameter"]

    if idx == -1:
      break
    
    divide(vectors, clusters, distinguishing_features, clusters_info, idx, whole_distances, k, len_vectors)
    try:
      score = silhouette_score(whole_distances, clusters, metric='precomputed')
    except:
      score = 0

    mod_clusters = [np.where(clusters == i)[0] for i in np.unique(clusters)]
    modularity_score = modularity(mod_clusters, whole_distances, m)

    myLogger.info("Modularity score at iteration %d: %f" % (i, modularity_score))
    myLogger.info("Silhouette_score at iteration %d: %f" % (i, score))

  return clusters, distinguishing_features, vectors, clusters_info

# divide the cluster into two clusters
def divide(vectors, clusters, distinguishing_features, clusters_info, cluster_id, whole_distances, k, len_vectors):

  myLogger.info('Dividing cluster %d' % cluster_id)

  distinguishing_feature = [f for idx, features in distinguishing_features[cluster_id] for f, score in features]
  selected_vector_idx = np.where(clusters == cluster_id)[0]
  # print(selected_vector_idx)
  selected_vector_size = len(selected_vector_idx)
  selected_vectors = vectors[selected_vector_idx].copy()
  # selected_vectors[:, distinguishing_features[cluster_id]] = 0
  selected_vectors[:, distinguishing_feature] = 0 # remove the distinguishing feature from the selected vectors to avoid the influence of the feature on the clustering result

  # print(distinguishing_features[cluster_id])

  distances = np.zeros((selected_vector_size, selected_vector_size))

  # compute the distances between the selected vectors without the distinguishing features
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
          # print(sequences[i], sequences[j])
          print(vectors[i], vectors[j])
          print(selected_vectors[i], selected_vectors[j])
          # print([concat_set_dict[n][idx] for idx in distinguishing_features[cluster_id]])
          # print([concat_set_dict[n][idx] for idx in distinguishing_feature])



  # splinters = [np.argmax(np.sum(distances, axis = 0))]

  mask = np.ones(selected_vector_size, bool)

  mask[np.argmax(np.average(distances, axis = 1))] = False
  # print(np.argmax(np.average(distances, axis = 1)))

  # select sessions that are farthest from the rest of the sessions
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

  # if divided clusters are too small, they are leaf clusters and we can stop
  if (in_clusters.size < max(min(MIN_CLUSTER_SIZE, 0.1 * len_vectors), 2)) or (out_clusters.size < max(min(MIN_CLUSTER_SIZE, 0.1 * len_vectors), 2)):
    clusters_info[cluster_id]['type'] = 'Leaf'
    return

  # assign unique id to the new clusters
  in_cluster_id = max(clusters_info.keys()) + 1
  out_cluster_id = max(clusters_info.keys()) + 2

  # update the clusters_info to store the new clusters

  clusters_info[cluster_id]["in_cluster_id"] = in_cluster_id
  clusters_info[cluster_id]["out_cluster_id"] = out_cluster_id

  clusters_info[cluster_id]["children"] = [in_cluster_id, out_cluster_id]
  
  # clusters_info[cluster_id]["children"].append(in_cluster_id)

  clusters[in_clusters] = in_cluster_id
  clusters[out_clusters] = out_cluster_id

  # contingency_table = np.asarray([np.sum(splinter_cluster, axis=0), np.sum(remaining_cluster, axis=0)])

  # for each cluster, compute the distinguishing features by selecting the features that are the frequently occurring in the inner cluster but not in the outer cluster with chi2 score
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

  # update the distinguishing_features to store the new distinguishing features

  distinguishing_features[in_cluster_id] = distinguishing_features[cluster_id] + [(in_cluster_id, cutoff_features)]
  distinguishing_features[out_cluster_id] = distinguishing_features[cluster_id] # + cutoff_features

  # compute the diameters of the new clusters
  in_cluster_dist = whole_distances[np.ix_(in_clusters, in_clusters)]
  out_cluster_dist = whole_distances[np.ix_(out_clusters, out_clusters)]


  clusters_info[in_cluster_id] = {
    "type": 'Tree',
    "cluster_size": in_clusters.shape[0],
    "diameter": np.max(in_cluster_dist),
    "children": None,
    "parent": None
  }


  clusters_info[out_cluster_id] = {
    "type": 'Tree',
    "cluster_size": out_clusters.shape[0],
    "diameter": np.max(out_cluster_dist),
    "children": None,
    "parent": None
  }

# %%



# %%
# group_by_sessions = queries.groupby('AnonID', sort=False, as_index=False)
# qqq = group_by_sessions.max(numeric_only=True)
# newIDs = qqq.loc[qqq['SessionNum'] >= 19]

# # %%
# group_by_sessions = queries.groupby(["AnonID", "SessionNum"])

# group_by_sessions = group_by_sessions.filter(lambda x: not ((
#   x["Query"].str.contains('porn|fuck|nude|sex|www|\.com|\.net', regex=True).any()
# )))

# group_by_sessions = group_by_sessions.groupby(["AnonID", "SessionNum"])


# tuples = []
# # for idx, row in newIDs.iterrows():
# #   tuples += [(row["AnonID"], i) for i in range(row["SessionNum"])]

# ## First, extract all tuples with appropriately long sessions

# # ssss = group_by_sessions.count()
# # ss = ssss # [ssss["Query"] >= 5]

# ssss = group_by_sessions.nunique()
# ss = ssss[ssss["Query"] <= 5]

# for idx, _ in ss.iterrows():
#   tuples += [idx]

# # %%
# ### Then, draw 5,000 samples

# SAMPLE_SIZE = 10000

# random.seed(3)

# sample = random.sample(tuples, SAMPLE_SIZE)

# sequences = []
# json_seqs = []
# query_text = []

# for s in sample:
#   g = group_by_sessions.get_group(s)
#   flattened_log, value, json_seq = flatten_logs(g)
#   queries = g[g['Type'] == "Query"]['Query']
#   v = []
#   queries_concat = queries.str.cat(sep='. ')
#   query_text.append(queries_concat)
#   sequences.append(flattened_log)
#   json_seqs.append({
#     "SessionNum": s[1],
#     "UserID": s[0],
#     "ClusterID": 0,
#     "Sequence": json_seq,
#     "pSkip": value[0],
#     "Click@1-5": value[1],
#     "MaxRR": value[2],
#     "MeanRR": value[3],
#     "AbandonmentRate": value[4],
#     "ReformulationRate": value[5],
#     "QueriesCount": value[6],
#     "ClicksPerQuery": value[7],
#     "BERTopicsKeywordCluster": 0,
#     "KMeansCluster": 0
#   })

# # %% KMeans Topic Clustering parts
# N_CLUSTERS = 100

# def c_tf_idf(documents, m, ngram_range=(1, 3)):
#     count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
#     t = count.transform(documents).toarray()
#     w = t.sum(axis=1)
#     tf = np.divide(t.T, w)
#     sum_t = t.sum(axis=0)
#     idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
#     tf_idf = np.multiply(tf, idf)

#     return tf_idf, count

# def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
#     words = count.get_feature_names()
#     tf_idf_transposed = tf_idf.T
#     indices = tf_idf_transposed.argsort()[:, -n:]
#     labels = list(range(N_CLUSTERS))
#     top_n_words = {}
#     for i, label in enumerate(labels):
#         # print(i)
#         top_n_words[label] = [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1]
# #     top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
#     return top_n_words

# # cc = KMeans(n_clusters=N_CLUSTERS)

# # query_embeddings = model.encode(query_text, convert_to_numpy=True)
# # query_embeddings = query_embeddings.astype('float64')
# # kmeans_labels = cc.fit_predict(query_embeddings)

# # docs = []

# # for i in range(N_CLUSTERS):
# #     idxs = np.argwhere(kmeans_labels == i)
# #     doc = ''
# #     for j in idxs:
# #         doc += (query_text[j[0]] + ' ')
# #     docs.append(doc)
  
# # tf_idf, count = c_tf_idf(docs, m=SAMPLE_SIZE)
# # top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs)

# # %% BERTopic Clustering parts

# topic_model = BERTopic(embedding_model = model, nr_topics="auto")
# topics, prob = topic_model.fit_transform(query_text)
# topics = np.asarray(topics)
# all_topics = topic_model.get_topics()

# %% Output 

# for i in range(len(topics)):
#   json_seqs[i]['BERTopicsKeywordCluster'] = int(topics[i])
# #  json_seqs[i]['KMeansCluster'] = int(kmeans_labels[i])

# topics, prob = topic_model.fit_transform(query_text)

# load data from create_keyword_clusters.py

with open('BERTopics-cluster-50000-50-new.json', 'r') as f:
  all_topics = json.load(f)

# with open('KMeans-clusters.json', 'w') as f:
#   json.dump(top_n_words, f, ensure_ascii=True, indent=2)

with open('sequences-50000-new.json', 'r') as f:
  json_seqs = json.load(f)

# %% 

# Sample from sequences

SAMPLE_SIZE = 7500 # number of sessions to sample
OUTLIER_SIZE = 500 #int(SAMPLE_SIZE * 0.1) # number of outliers (sessions without any topic) to sample
VALID_SIZE = SAMPLE_SIZE - OUTLIER_SIZE

valid_cluster_ids = [1, 2, 3, 4, 5, 6, 7, 9, 14, 16, 17, 19, 20, 22, 24, 25, 27, 28, 31, 33, 35, 36, 38, 39, 40, 41, 42, 43, 45, 47, 48, 49, 50, 52, 53, 55, 56, 58, 59, 61, 65, 68, 69, 74, 76, 79] # list of user query cluster ids used in the dashboard

# sequences = flatten_logs(json_seqs)

outliers = [s for s in json_seqs if s['BERTopicsKeywordCluster'] == -1]
valid_sequences = [s for s in json_seqs if s['BERTopicsKeywordCluster'] in valid_cluster_ids]
random.seed(3)
random_outliers = random.sample(outliers, k=OUTLIER_SIZE)
random_valids = random.sample(valid_sequences, k=VALID_SIZE)

random_seqs = random_outliers + random_valids
sequences = flatten_logs(random_seqs)

topics = np.asarray([s['BERTopicsKeywordCluster'] for s in random_seqs])

# %%
seq_dict = {}
ngram_dict = {}
concat_set_dict = {}
clusters_dict = {}
distinguishing_features_dict = {}
vectors_dict = {}
clusters_info_dict = {}


# %%

for k in [20]: # adjust this to change the number of maximum number of distinguishing features
  for n in range(3, 4): # adjust this to change the length of ngrams
    
    ngram_dict[n], concat_set_dict[n] = generate_n_grams(sequences, n = n)
    for j in set(topics):
      idxs = np.argwhere(topics == int(j)).flatten()
      clusters_dict[n], distinguishing_features_dict[n], vectors_dict[n], clusters_info_dict[n] = divisive_clustering(ngram_dict[n], concat_set_dict[n], 100, k, target = idxs)
      try:
        score = silhouette_score(vectors_dict[n], clusters_dict[n], metric='cosine')
        myLogger.info("n-gram size: %f, silhouette score: %f" % (n, score))
      except:
        myLogger.info("n-gram size: %f, silhouette score: %f" % (n, 0))


      merge_clusters(clusters_info_dict[n], distinguishing_features_dict[n], clusters_dict[n], vectors_dict[n])

      # with open(f'n-gram-{n}-{k}.txt', 'a') as f:
      #   f.write(','.join(['clusterId', 'sequences', 'userId', 'sessionNum', 'initialQuery']))
      #   f.write('\n')
      #   for idx, i in enumerate(idxs):
      #     label_divisive = str(clusters_dict[n][idx])
      #     userid = str(sample[i][0])
      #     group = str(sample[i][1])
      #     g = group_by_sessions.get_group(sample[i])
      #     previous_query = g.iloc[0]['Query']
      #     row = label_divisive + ',' + '+'.join(sequences[idx]) + ',' + userid + ',' + group + ',' + previous_query + '\n'
      #     f.write(row)
      
      # with open(f'cluster-info-{n}-{k}.txt', 'w') as f:
      #   for key, cluster_info in clusters_info_dict[n].items():
      #     f.write("Cluster %d: diameter %f, size %d, type %s \n" % (key, cluster_info["diameter"], cluster_info["cluster_size"], cluster_info["type"]))
      #     children = str(cluster_info["children"])
      #     f.write(f"Children {children} \n")
      #     f.write("Distinguishing factors\n")
      #     f.write(str([concat_set_dict[n][idx] for idx in distinguishing_features_dict[n][key]]))
      #     f.write("\n")

      # save behavior clusters as json file

      with open(f'cluster-info-{n}-{k}.json', 'a') as f:
        tree = {
          "root_id": 1,
          "nodes": [],
          "keyword_cluster": int(j)
        }
        for key, cluster_info in clusters_info_dict[n].items():
          distinguishing_feature = [(cluster_id, f, score) for cluster_id, features in distinguishing_features_dict[n][key] for f, score in features]
          children = cluster_info["children"]
          if children is not None:
            for child in children:
              clusters_info_dict[n][child]['parent'] = key
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
            "subtree_size": cluster_info["cluster_size"],
            "parent": cluster_info['parent'] # TODO
          }
          tree["nodes"].append(node)

        json.dump(tree, f, ensure_ascii=True, indent = 2)
      for idx, i in enumerate(idxs):
        label_divisive = clusters_dict[n][idx]
        random_seqs[i]["ClusterID"] = int(label_divisive)

    # save the session information as json file
    with open(f'sequences-{n}-{k}.json', 'w') as f:
      json.dump(random_seqs, f, ensure_ascii=True, indent = 2)

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


