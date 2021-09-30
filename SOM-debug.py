# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import csv
from numpy.random.mtrand import beta
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sn
import random
from scipy import spatial
from bert_serving.client import BertClient
from scipy.special import logsumexp
plt.close('all')
# bc = BertClient(ip="internal.kixlab.org", port=8888, port_out=8889)
# old = np.seterr(all='raise')

# %%
queries = pd.read_csv('new_logs.csv', index_col="idx", dtype={"AnonID": "Int64", "Query": "string", "QueryTime": "string", "ItemRank": "Int32", "ClickURL": "string", "Type": "string", "SessionNum": "Int32"}, keep_default_na=False, na_values=[""])


# %%
group_by_sessions = queries.groupby('AnonID', sort=False, as_index=False)
qqq = group_by_sessions.max(numeric_only=True)
newIDs = qqq.loc[qqq['SessionNum'] >= 19]
# print(newIDs)
# print(newIDs["AnonID"])


# %%

def compute_edit_distance(queryA, queryB):
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

def compute_semantic_similarity(old_query, new_query):
  # embeddings = bc.encode([old_query, new_query])
  # return 1 - spatial.distance.cosine(embeddings[0], embeddings[1])

  return 1

def is_new_query(old_query, new_query):
  # compute semantic similarities and edit distances to gauge query similarities
  if compute_edit_distance(old_query, new_query) < 5: #
    return False

  if compute_semantic_similarity(old_query, new_query) > 0.7:
    return False
    
  return True

def flatten_logs(logs_dataframe):
  previous_row = None
  previous_query = ''
  flattened = []
  
  for idx, row in logs_dataframe.iterrows():
    if previous_row is None: # Every logstream starts with a query
      # print(row)
      flattened.append({
        "type": "NewQuery",
        "query": row['Query']
      })
      previous_row = row
      previous_query = row['Query']
      
      continue

    if row["Type"] == "Query":
      if is_new_query(previous_query, row['Query']):
        flattened.append({
          "type": "NewQuery",
          "query": row['Query']
        })
      else:
        flattened.append({
          "type": "RefinedQuery", # should be RefinedQuery
          "query": row['Query']
        })
    elif row["Type"] == "Click":
      if row["ItemRank"] == 1:
        flattened.append({
          "type": "Click1",
          "rank": row['ItemRank']
        })
      elif row["ItemRank"] >= 2 and row["ItemRank"] < 6:
        flattened.append({
          "type": "Click2-5",
          "rank": row['ItemRank']
        })
      elif row["ItemRank"] >= 6 and row["ItemRank"] < 10:
        flattened.append({
          "type": "Click6-10",
          "rank": row['ItemRank']
        })
      else:
        flattened.append({
          "type": "Click11+",
          "rank": row['ItemRank']
        })
    elif row["Type"] == "NextPage":
      flattened.append({
        "type": "NextPage"
      })

  return flattened


# %%
group_by_sessions = queries.groupby(["AnonID", "SessionNum"])

tuples = []
# for idx, row in newIDs.iterrows():
#   tuples += [(row["AnonID"], i) for i in range(row["SessionNum"])]

## First, extract all tuples with appropriately long sessions

ssss = group_by_sessions.count()
ss = ssss[ssss["Query"] <= 10]
ss = ss[ss["Query"] >= 3]

for idx, _ in ss.iterrows():
  tuples += [idx]

### Then, draw 5,000 samples

SAMPLE_SIZE = 1000

random.seed(0)
sample = random.sample(tuples, SAMPLE_SIZE)


# %%
# Markov Model

class MarkovModel(object):
  states = np.ndarray(0)
  num_states = 0
  probabilities = None
  log_probabilities = None
  initial_probabilities = np.ndarray(0)
  x = 0
  y = 0

  def __init__(self, states, x, y, probabilities = None, initial_probabilities = None):
    self.states = states
    self.probabilities = probabilities
    self.initial_probabilities = initial_probabilities # This will be 1 and 0 anyways
    self.x = x
    self.y = y

    if self.probabilities is None:
      self.probabilities = np.random.rand(len(states), len(states))
      self.probabilities = self.probabilities / np.sum(self.probabilities, axis = 1)[:, None]
      self.log_probabilities = np.log(self.probabilities)

  def get_coordinates(self):
    return self.x, self.y

  def update_probabilities(self, probabilities):
    self.probabilities = probabilities
    # self.log_probabilities = np.log(self.probabilities)

  def update_log_probabilities(self, log_probabilities):
    self.log_probabilities = log_probabilities
    self.probabilities = np.exp(log_probabilities)

  # def compute_probability(self, sequence):
  #   prob = 1
  #   prev_state_idx = -1
  #   for idx, state in enumerate(sequence):
  #     cur_state_idx = self.states.index(state['type'])
  #     if idx == 0: # initial prob
  #       prob = prob * self.initial_probabilities[cur_state_idx]
  #       prev_state_idx = cur_state_idx
  #     else:
  #       prob = prob * self.probabilities[prev_state_idx, cur_state_idx]
  #       prev_state_idx = cur_state_idx

    
  #   return prob

  def compute_probability(self, sequence):
    return np.exp(self.compute_log_probability(sequence))

  def compute_log_probability(self, sequence):
    log_prob = 0
    prev_state_idx = -1
    for idx, state in enumerate(sequence):
      cur_state_idx = self.states.index(state['type'])
      if idx == 0: # initial prob
        log_prob = log_prob + np.log(self.initial_probabilities[cur_state_idx])
        prev_state_idx = cur_state_idx
      else:
        log_prob = log_prob + np.log(self.probabilities[prev_state_idx, cur_state_idx])
        prev_state_idx = cur_state_idx

    
    return log_prob



# %%
# https://lovit.github.io/visualization/2019/12/02/som_part1/

def initialize_simple(n_rows, n_cols):

  grid, pairs = make_grid_and_neighbors(n_rows, n_cols)

  x_ranges = np.linspace(0, 1, n_rows)
  y_ranges = np.linspace(0, 1, n_cols)

  C = np.asarray([[x, y] for x in x_ranges for y in y_ranges])

  return grid, C, pairs

def initialize_markov(n_rows, n_cols, states, initial_probabilities):

  grid, pairs = make_grid_and_neighbors(n_rows, n_cols)

  x_ranges = np.linspace(-1, 1, n_rows)
  y_ranges = np.linspace(-1, 1, n_cols)

  C = np.asarray([MarkovModel(states, x, y, None, initial_probabilities) for x in x_ranges for y in y_ranges])


  return grid, C, pairs

def make_masks_markov(grid, sigma = 1.0, max_width = 2):
  rows, cols = np.where(grid >= 0)
  data = grid[rows, cols]

  sorted_indices = data.argsort()
  indices = zip(rows[sorted_indices], cols[sorted_indices])
  masks = [make_gaussian_mask_markov(grid, i, j, sigma, max_width) for i, j in indices]
  masks = [mask.flatten() for mask in masks]

  return masks

def make_gaussian_mask_markov(grid, i, j, sigma = 1.0, max_width = 2):
  mask = np.zeros(grid.shape)
  center = C[grid[i, j]]
  y = center.y
  x = center.x
  for i_, j_ in zip(*np.where(grid >= 0)):
    if (max_width > 0) and (abs(i - i_) + abs(j - j_) > max_width):
      continue
    target = C[grid[i_, j_]]
    x_ = target.x 
    y_ = target.y
    mask[i_, j_] = np.exp(- ((i - i_) ** 2 + (j - j_) ** 2) / (sigma ** 2)) # / (np.sqrt(np.pi * 2) * sigma) 

  return mask


def make_grid_and_neighbors(n_rows, n_cols):
  grid = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
  pairs = []
  for i in range(n_rows):
    for j in range(n_cols):
      idx = grid[i, j]
      neighbors = []
      if j > 0:
        neighbors.append(grid[i, j-1])
      if i > 0:
        neighbors.append(grid[i-1, j])
      for nidx in neighbors:
        pairs.append((idx, nidx))

  return grid, pairs

def make_masks(grid, sigma = 1.0, max_width = 2):
  rows, cols = np.where(grid >= 0)
  data = grid[rows, cols]

  sorted_indices = data.argsort()
  indices = zip(rows[sorted_indices], cols[sorted_indices])
  masks = [make_gaussian_mask(grid, i, j, sigma, max_width) for i, j in indices]
  masks = [mask.flatten() for mask in masks]

  return masks

def make_gaussian_mask(grid, i, j, sigma = 1.0, max_width = 2):
  mask = np.zeros(grid.shape)
  for i_, j_ in zip(*np.where(grid >= 0)):
    if (max_width > 0) and (abs(i - i_) + abs(j - j_) > max_width):
      continue
    mask[i_, j_] = np.exp(-((i-i_)**2 + (j-j_) ** 2) / sigma ** 2) / (sigma)

  return mask

def make_neighbor_graph(grid, max_width = 2, decay = 0.25):
  def weight_array(f, s):
    return np.asarray([np.power(f, i) for i in range(1, s+1) for _ in range(4*i)])

  def pertubate(s):
    def unique(i, s):
      if abs(i) == s:
        return [0]
      return [s - abs(i), -s + abs(i)]

    def pertubate_(s_):
      return [(i, j) for i in range(-s, s+1) for j in unique(i, s_)]

    return [pair for s_ in range(1, s+1) for pair in pertubate_(s_)]

  def is_outbound(i_, j_):
    return (i_ < 0) or (i_ >= n_rows) or (j_ < 0) or (j_ >= n_cols)

  n_rows, n_cols = grid.shape
  n_codes = n_rows * n_cols

  W = weight_array(decay, max_width)
  N = -np.ones((n_codes, W.shape[0]), dtype = np.int)
  N_inv = -np.ones((n_codes, W.shape[0]), dtype = np.int)

  for row, (i, j) in enumerate(zip(*np.where(grid >= 0))):
    idx_b = grid[i, j]
    for col, (ip, jp) in enumerate(pertubate(max_width)):
      if is_outbound(i+ip, j+jp):
        continue
      idx_n = grid[i+ip, j+jp]
      N[idx_b, col] = idx_n
      N_inv[idx_n, col] = idx_b
  
  return N, N_inv, W


# %%
from sklearn.metrics import pairwise_distances_argmin_min
def closest(X, C, metric):
  # return (idx, dist)

  return pairwise_distances_argmin_min(X, C, metric = metric)

def update_stochastic(X, C, lr = 0.01, metric = 'euclidean', masks = None):
  n_data = X.shape[0]
  n_codes, n_features = C.shape
  C_new = C.copy()

  Xr = X[np.random.permutation(n_data)]

  for i, Xi in enumerate(Xr):
    bmu, _ = closest(Xi.reshape(1, -1), C_new, metric)
    bmu = int(bmu)

    diff = Xi - C_new
    grad = lr * diff * masks[bmu][:, np.newaxis]
    C_new += grad
  
  return C_new
    

# def update_cmeans(X, C, update_ratio, metric='euclidean', batch_size = -1, grid = None, neighbors = None, inv_neighbors = None, weights = None, adjust_ratio = 0.5, max_width = 2, decay = 0.25, **kargs):
#   if (neighbors in None) or (weights is None):
#     neighbors, inv_neighbors, weights = make_neighbor_graph(grid, max_width, decay)

#   C_new = C.copy()

#   for b, Xb in enumerate(to_minibatch(X, batch_size)):
#     C_new = update_cmeans_batch(Xb, C_new, update_ratio, metric, neighbors, inv_neighbors, weights, adjust_ratio)

#   return C_new

def update_cmeans_batch(X, C, update_ratio, metric, neighbors, inv_neighbors, weights, adjust_ratio):
  n_data = X.shape[0]
  n_codes = C.shape[0]

  C_cont = np.zeros(shape = C.shape)
  W_new = np.zeros(n_codes)

  bmu, dist = closest(X, C, metric)

  for bmu_c in range(n_codes):
    indices = np.where(bmu == bmu_c)[0]
    n_matched = indices.shape[0]

    if n_matched == 0:
      continue
      
    Xc = np.asarray(X[indices, :].sum(axis=0)).reshape(-1)
    C_cont[bmu_c] += Xc
    W_new[bmu_c] += n_matched

    if weights.shape[0] == 0:
      continue

    for c, w in zip(neighbors[bmu_c], weights):
      if c == -1:
        continue
      C_cont[c] += w * Xc
      W_new[c] += w * n_matched

  C_new = update_ratio * C_cont + (1 - update_ratio) * C
  return C_new


# %%
# Markov model specific implementations here

def find_pair_from_sequence(seq, state_prev, state_next):
  for i in range(len(seq) - 1):
    if seq[i]['type'] == state_prev and seq[i+1]['type'] == state_next:
      return 1
    
  return 0

def logdot(a, b):
  max_a, max_b = np.max(a), np.max(b)
  exp_a, exp_b = a - max_a, b - max_b
  np.exp(exp_a, out=exp_a)
  np.exp(exp_b, out=exp_b)
  c = np.dot(exp_a, exp_b)
  np.log(c, out=c)
  c += max_a + max_b
  return c

def logdivide(a, b):
  log_a, log_b = np.log(a), np.log(b)
  max_a = max(np.max(log_a), np.max(log_b))
  exp_a, exp_b = log_a - max_a, log_b - max_a
  np.exp(exp_a, out=exp_a)
  np.exp(exp_b, out=exp_b)
  c = exp_a / exp_b

  return c

def update_markov(sequences, C, states, grid, min_neighborhood_size = 0, decreasing_factor = 0.97, epochs = 100):
  eps = np.finfo(np.float64).eps
  K = C.size # grid size
  C_flat = C.ravel()
  N = len(sequences) # number of inputs
  m = len(states)

  delta_matrix = np.asarray([[1 if sequences[_n][0]['type'] == states[_m] else 0 for _n in range(N)] for _m in range(m)] ) # m X N deltas
  beta_matrix = np.asarray([[[find_pair_from_sequence(sequences[_n], states[_prev], states[_next]) for _n in range(N)] for _next in range(m)] for _prev in range(m)]) # m X m X N matrix of betas

  cur_neighborhood_size = 1.5

  prev_likelihood = - np.Inf

  # E-step: compute Q function and update probabilities

  for i_ in range(epochs):

    # Compute and print likelihood
    # old = np.seterr(all='raise')

    masks = make_masks_markov(grid, sigma = cur_neighborhood_size, max_width = 5) # h-matrix, with row vectors as mask for each grid point
    # delta_matrix = np.asarray([[1 if sequences[_n][1]['type'] == states[_m] else 0 for _n in range(N)] for _m in range(m)] ) # m X N deltas
    # beta_matrix = np.asarray([[[find_pair_from_sequence(sequences[_n], states[_prev], states[_next]) for _n in range(N)]for _prev in range(m)] for _next in range(m)]) # m X m X N matrix of betas

    p = np.asarray([[C[k].compute_log_probability(sequences[n]) for k in range(K)] for n in range(N)]) # N by K matrix with log probs
    p_c = np.zeros((N, K)) # log(p_c), N by K matrix 

    for n in range(N):
      for k in range(K):
        mask = masks[k]
        p_c[n,k] = np.nan_to_num(np.dot(p[n, :], mask), nan=0, posinf=np.inf, neginf = -np.inf)

        # for r in np.where(mask > 0)[0]:
          # p_c[n, k] += (p[n, r] * mask[r])

          # if r == k:
          #   p_c[n, k] += p[n, k]
          # else:
          #   p_c[n, k] += (p[n, r] * mask[r])
    
    # np.seterr(**old)
    likelihood = 0
    for n in range(N):
      # _sum = 0
      # for k in range(K):
        # _sum = logdot(_sum, p_c[n, k])
      # _sum = np.log(np.sum(np.exp(p_c[n, :])) / K)
      _sum = logsumexp((p_c[n, :]))
      likelihood += _sum

    likelihood -= np.log(K) * N

    # old = np.seterr(all = 'raise')

    # if (likelihood < prev_likelihood) or np.isnan(likelihood) or np.isinf(likelihood):
    if np.isnan(likelihood) or np.isinf(likelihood):
      break
    
    prev_likelihood = likelihood



    # M-step: update params

    # initial states

    for k in range(K):
      # np.seterr(**old)

      # masked_pc = np.exp(logdot(p_c[:, k, None], np.log(masks[k][None, :])))
      masked_pc = logdot(p_c[:, k, None], np.log(masks[k][None, :]))
      # old = np.seterr(all='raise')
      
      denom_matrix = logdot(np.log(delta_matrix), masked_pc) # ending up with (m X N) (N X 1) (1 X K) == m X K 
      denom_vector = logsumexp(denom_matrix, axis=1)
      new_initials = np.exp(denom_vector - logsumexp(denom_vector))
      # denom_matrix = delta_matrix @ masked_pc # ending up with (m X N) (N X 1) (1 X K) == m X K 
      # denom_vector = np.sum(denom_matrix, axis = 1)

      # new_initials = denom_vector / np.linalg.norm(denom_vector, ord = 1) # denom_vector / np.sum(denom_vector)
      C_flat[k].initial_probabilities = new_initials

      # denom_ndarray = beta_matrix @ masked_pc # ending up with (m, m, N) (N, 1) (1, K) == m, m, K
      # denom_mat = np.sum(denom_ndarray, axis = 2)
      # new_transitions = denom_mat / (np.linalg.norm(denom_mat, ord = 1, axis = 1)[:, None])

      denom_ndarray = logdot(np.log(beta_matrix), masked_pc)
      denom_mat = logsumexp(denom_ndarray, axis=2)
      new_transitions = denom_mat - logsumexp(denom_mat, axis=1)[:, None]

      # new_transitions = denom_mat / np.sum(denom_mat, axis = 1)[:, None]
      
      # np.seterr(**old)
      # new_transitions = logdivide(denom_mat, np.sum(denom_mat, axis = 1)[:, None])
      # old = np.seterr(all='raise')

 
      C_flat[k].update_log_probabilities(new_transitions)

    # # initial states -- log version
    # for k in range(K):
    #   new_initials = np.zeros(m)
    #   for r in range(m):
    #     val = 0
    #     for n in range(N):
    #       for l in range(K):
    #         val += np.exp(p_c[n, k] + np.log(masks[k][l])) * delta_matrix[r, n]
    #     new_initials[r] = val

    #   new_initials = new_initials / np.sum(new_initials)
    #   C_flat[k].initial_probabilities = new_initials

    # Transition states -- log version

    # for k in range(K):
    #   new_transitions = np.zeros((m,m))

    #   for r in range(m): # previous state
    #     for s in range(m): # current state
    #       val = 0
    #       for n in range(N):
    #         for l in range(K):
    #           val += np.exp(p_c[n, k] + np.log(masks[k][l])) * beta_matrix[r, s, n]
    #       new_transitions[r, s] = val
        
    #   new_transitions = new_transitions / np.sum(new_transitions, axis = 1)[:, None]
    #   C_flat[k].update_probabilities(new_transitions)



    # Transition states

    # for k in range(K):
    #   denom_ndarray = beta_matrix @ masked_pc # ending up with (m, m, N) (N, 1) (1, K) == m, m, K
    #   denom_mat = np.sum(denom_ndarray, axis = 2)
    #   # if np.any(denom_mat < eps * 100):
    #   #   denom_mat = denom_mat * (10 ** 10)
    #   # new_transitions = denom_mat / np.linalg.norm(denom_mat, ord = 1, axis = 1)
    #   new_transitions = denom_mat / np.sum(denom_mat, axis = 1)[:, None]
 
    #   C_flat[k].update_probabilities(new_transitions)






    if (cur_neighborhood_size < min_neighborhood_size):
      break
    

    cur_neighborhood_size = cur_neighborhood_size * decreasing_factor

    print("Current Likelihood: %f" % likelihood)

    if i_ % 10 == 5:
      coordinates = plot_clickstreams(sequences, C)
      sn.scatterplot(x = coordinates[:, 0], y = coordinates[:, 1])
      plt.show(block = False)
    
    # np.seterr(**old)

  
# %%

def plot_clickstreams(sequences, C):

  N = len(sequences)
  K = C.size

  coordinates = np.zeros((N, 2))
  x_vector = np.asarray([C[k].x for k in range(K)])
  y_vector = np.asarray([C[k].y for k in range(K)])
  for n in range(N):
    prob_vector = np.asarray([C[k].compute_log_probability(sequences[n]) for k in range(K)])
    prob_vector = prob_vector - logsumexp(prob_vector)
    prob_vector = np.exp(prob_vector)

    coordinates[n,0] = np.dot(x_vector, prob_vector)
    coordinates[n,1] = np.dot(y_vector, prob_vector)

  return coordinates
  


# %%
sequences = []

for s in sample:
  g = group_by_sessions.get_group(s)
  sequences.append(flatten_logs(g))

states = ["NewQuery", "RefinedQuery", "Click1", "Click2-5", "Click6-10", "Click11+", "NextPage"]

grid, C, pairs = initialize_markov(21, 21, states, [1] + [0 for i in range(len(states) - 1)])

coordinates = plot_clickstreams(sequences, C)

sn.scatterplot(x = coordinates[:, 0], y = coordinates[:, 1])

plt.show(block = False)

# %%

update_markov(sequences, C, states, grid, min_neighborhood_size = 0.2, epochs = 30)


import pickle

with open('C.pkl', 'wb') as fw:
  pickle.dump(C, fw)

with open('seq.pkl', 'wb') as fw:
  pickle.dump(sequences, fw)


# %%

# with open('C.pkl', 'rb') as fw:
#   C = pickle.load(fw)

# with open('seq.pkl', 'rb') as fw:
#   sequences = pickle.load(fw)

coordinates = plot_clickstreams(sequences, C)

sn.scatterplot(x = coordinates[:, 0], y = coordinates[:, 1])

plt.show()

print('done')
