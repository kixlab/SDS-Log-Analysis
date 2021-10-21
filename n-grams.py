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
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import chisquare, chi2_contingency
from rhc import mutual_info
from rhc import recursiveHierarchicalClustering
import logging
from datetime import datetime

np.seterr(all='raise')

nlp = spacy.load('en_core_web_lg')
stopwords = nlp.Defaults.stop_words

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
  
  for idx, row in logs_dataframe.iterrows():
    if previous_row is None: # Every logstream starts with a query
      # print(row)
      flattened.append("NewQuery")
      previous_row = row
      previous_query = row['Query']
      previous_processed_query = process_query(previous_query)
      
      continue

    if row["Type"] == "Query":
      new_processed_query = process_query(row['Query'])
      if is_new_query(previous_query, previous_processed_query, row['Query'], new_processed_query):
        flattened.append("NewQuery")
        previous_query = row['Query']
        previous_processed_query = new_processed_query
      else:
        flattened.append("RefinedQuery")
    elif row["Type"] == "Click":
      if row["ItemRank"] == 1:
        flattened.append("Click1")
      elif row["ItemRank"] >= 2 and row["ItemRank"] < 6:
        flattened.append("Click2-5")
      elif row["ItemRank"] >= 6 and row["ItemRank"] < 10:
        flattened.append("Click6-10")
      else:
        flattened.append("Click11+")
    elif row["Type"] == "NextPage":
      flattened.append("NextPage")

  return flattened


def create_n_gram(sequence, n):
  if n < 2:
    return []
  ngram = []
  length = len(sequence)
  if length < n:
    filled = sequence + ["Empty" for i in range(n - length)]
    ngram.append(tuple(filled))

  else:
    for i in range(length - n + 1):
      ngram.append(tuple(sequence[i:i+n]))

  return ngram + create_n_gram(sequence, n - 1)


def compute_distance(n_gram_1, n_gram_2, concatenated_set):
  counter_ngram_1 = Counter(n_gram_1)
  counter_ngram_2 = Counter(n_gram_2)


  n_gram_vec_1 = np.asarray([(counter_ngram_1[n]) for n in concatenated_set])
  n_gram_vec_1 = n_gram_vec_1 / np.linalg.norm(n_gram_vec_1)
  n_gram_vec_2 = np.asarray([(counter_ngram_2[n]) for n in concatenated_set])
  n_gram_vec_2 = n_gram_vec_2 / np.linalg.norm(n_gram_vec_2)

  distance = np.arccos((np.dot(n_gram_vec_1, n_gram_vec_2) / np.sqrt(np.dot(n_gram_vec_1, n_gram_vec_1) * np.dot(n_gram_vec_2, n_gram_vec_2)))) / np.pi

  return distance

def ngrams_to_vectors(n_gram_1, concatenated_set):
  counter_ngram_1 = Counter(n_gram_1)
  n_gram_vec_1 = np.asarray([(counter_ngram_1[n]) for n in concatenated_set])
  # n_gram_vec_1 = n_gram_vec_1 / np.linalg.norm(n_gram_vec_1)

  return n_gram_vec_1


def generate_sequences_and_n_grams(group_by_sessions, sample, n = 5):
  sequences = []
  ngrams = []

  for s in sample:
    g = group_by_sessions.get_group(s)
    flattened_log = flatten_logs(g)
    ngram = create_n_gram(flattened_log, n)
    sequences.append(flattened_log)
    ngrams.append(ngram)

  concat_set = list(set([ngram for n in ngrams for ngram in n]))

  return sequences, ngrams, concat_set

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
      e_matrix[j, i] = e_matrix[i, j]

  modularity = np.trace(e_matrix) - np.sum(np.dot(e_matrix, e_matrix))

  # print(e_matrix)

  return modularity

def normalize_vectors(vectors):
  # average_vector = np.average(vectors, axis=0)
  normalized_vectors = vectors #  - average_vector

  normalized_vectors = normalize(normalized_vectors, axis = 1)

  return normalized_vectors

def compute_polar_distance(n_gram_vec_1, n_gram_vec_2):
  distance = np.arccos((np.dot(n_gram_vec_1, n_gram_vec_2) / np.sqrt(np.dot(n_gram_vec_1, n_gram_vec_1) * np.dot(n_gram_vec_2, n_gram_vec_2)))) / np.pi
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


  MIN_CLUSTER_SIZE = 100


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
    score = silhouette_score(vectors, clusters, metric='cosine')

    mod_clusters = [np.where(clusters == i)[0] for i in np.unique(clusters)]
    modularity_score = modularity(mod_clusters, whole_distances, m)

    myLogger.info("Modularity score at iteration %d: %f" % (i, modularity_score))
    myLogger.info("Silhouette_score at iteration %d: %f" % (i, score))

  return clusters, distinguishing_features, vectors, clusters_info

def divide(vectors, clusters, distinguishing_features, clusters_info, cluster_id, whole_distances, k):

  myLogger.info('Dividing cluster %d' % cluster_id)
  selected_vector_idx = np.where(clusters == cluster_id)[0]
  # print(selected_vector_idx)
  selected_vector_size = len(selected_vector_idx)
  selected_vectors = vectors[selected_vector_idx].copy()
  selected_vectors[:, distinguishing_features[cluster_id]] = 0
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
          print([concat_set_dict[n][idx] for idx in distinguishing_features[cluster_id]])


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

  in_cluster_id = max(clusters_info.keys()) + 1
  out_cluster_id = max(clusters_info.keys()) + 2

  clusters_info[cluster_id]["children"] = [in_cluster_id, out_cluster_id]

  in_clusters = selected_vector_idx[~mask]
  out_clusters = selected_vector_idx[mask]

  clusters[in_clusters] = in_cluster_id
  clusters[out_clusters] = out_cluster_id

  # contingency_table = np.asarray([np.sum(splinter_cluster, axis=0), np.sum(remaining_cluster, axis=0)])

  max_chisq = 0
  max_idx = 0
  chisqs = []
  for i in range(vectors.shape[1]):
    if i in distinguishing_features[cluster_id]:
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
  cutoff_features = [x[2] for x in chisqs[:(res+1)]]
  # print(chisqs, res, cutoff_features)
  # cutoff_point = next(i for i,v in enumerate(chisqs) if v[0] == res)
  # cutoff_features = [x[0] for x in chisqs[:cutoff_point]]
  # print(chisqs)
  # print(res)
  # print(cutoff_features)

  distinguishing_features[in_cluster_id] = distinguishing_features[cluster_id] + cutoff_features
  distinguishing_features[out_cluster_id] = distinguishing_features[cluster_id] + cutoff_features

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
group_by_sessions = queries.groupby('AnonID', sort=False, as_index=False)
qqq = group_by_sessions.max(numeric_only=True)
newIDs = qqq.loc[qqq['SessionNum'] >= 19]

# %%
group_by_sessions = queries.groupby(["AnonID", "SessionNum"])

tuples = []
# for idx, row in newIDs.iterrows():
#   tuples += [(row["AnonID"], i) for i in range(row["SessionNum"])]

## First, extract all tuples with appropriately long sessions

ssss = group_by_sessions.count()
ss = ssss[ssss["Query"] >= 5]

for idx, _ in ss.iterrows():
  tuples += [idx]

# %%
### Then, draw 5,000 samples

SAMPLE_SIZE = 5000

random.seed(10)

sample = random.sample(tuples, SAMPLE_SIZE)

sequences = []

for s in sample:
  g = group_by_sessions.get_group(s)
  flattened_log = flatten_logs(g)
  sequences.append(flattened_log)

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
  for n in range(2, 6):
    ngram_dict[n], concat_set_dict[n] = generate_n_grams(sequences, n = n)
    clusters_dict[n], distinguishing_features_dict[n], vectors_dict[n], clusters_info_dict[n] = divisive_clustering(ngram_dict[n], concat_set_dict[n], 10, k)
    score = silhouette_score(vectors_dict[n], clusters_dict[n], metric='cosine')
    myLogger.info("n-gram size: %f, silhouette score: %f" % (n, score))

    with open(f'n-gram-{n}-{k}.txt', 'w') as f:
      f.write(','.join(['clusterId', 'sequences', 'userId', 'sessionNum', 'initialQuery']))
      for i in range(SAMPLE_SIZE):
        label_divisive = str(clusters_dict[n][i])
        userid = str(sample[i][0])
        group = str(sample[i][1])
        g = group_by_sessions.get_group(sample[i])
        previous_query = g.iloc[0]['Query']
        row = label_divisive + ',' + '+'.join(sequences[i]) + ',' + userid + ',' + group + ',' + previous_query + '\n'
        f.write(row)
    
    with open(f'cluster-info-{n}-{k}.txt', 'w') as f:
      for key, cluster_info in clusters_info_dict[n].items():
        f.write("Cluster %d: diameter %f, size %d \n" % (key, cluster_info["diameter"], cluster_info["cluster_size"]))
        children = str(cluster_info["children"])
        f.write(f"Children {children} \n")
        f.write("Distinguishing factors\n")
        f.write(str([concat_set_dict[n][idx] for idx in distinguishing_features_dict[n][key]])) 

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


