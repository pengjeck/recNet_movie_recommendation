# -*- coding:utf-8 -*-

import wmf
import numpy as np
from scipy.sparse import csr_matrix
import pickle
from sklearn.decomposition import NMF

data = "m1m"

# with open("../" + data + "/pdata/vtr.pkl", "rb") as f:
# 	[visit, vtr, vte] = pickle.load(f, encoding="utf8")
#
# with open("../"+data + "/pdata/map.pkl", "rb") as f:
# 	[user_id, id_user, poi_id, id_poi] = pickle.load(f, encoding="utf8")
#
# with open("../" + data + "/pdata/ctr.pkl", "rb") as f:
# 	[ci, ctr, cte] = pickle.load(f, encoding="utf8")
#
# taboo = {}
# up_matrix = np.zeros((len(user_id), len(poi_id)), dtype=float)
# for user in ctr:
# 	uid = user_id[user]
# 	taboo[uid] = set()
# 	for ch in ctr[user]:
# 		pid = poi_id[ch[0]]
# 		up_matrix[uid, pid] += 1
# 		taboo[uid].add(pid)
# 	taboo[uid] = list(taboo[uid])
#
# test_cases = []
# for user in cte:
# 	uid = user_id[user]
# 	for ch in cte[user]:
# 		pid = poi_id[ch[0]]
# 		test_cases.append((uid, pid))
#
# with open("data/"+data+".pkl", "wb") as f:
# 	pickle.dump([up_matrix, taboo, test_cases], f, protocol=2)

with open("../data/"+data+"/data.pkl", "rb") as f:
	[mtr, mte] = pickle.load(f)

with open("../data/"+data+"/maps.pkl", "rb") as f:
	[user_id, id_user, movie_id, id_movie] = pickle.load(f)

taboo = mtr
up_matrix = np.zeros((len(user_id), len(movie_id)), dtype=float)
for uid in mtr:
	for mid in mtr[uid]:
		up_matrix[uid, mid] += 1

test_cases = []
for uid in mte:
	for mid in mte[uid]:
		test_cases.append((uid, mid))

gr = {}
for case in test_cases:
	uid, pid = case
	if uid in gr:
		gr[uid].add(pid)
	else:
		gr[uid] = {pid}

C = csr_matrix(up_matrix)

S = wmf.log_surplus_confidence_matrix(C, alpha=2.0, epsilon=1e-6)
U, V = wmf.factorize(S, num_factors=100, lambda_reg=1e-2, num_iterations=10, init_std=0.01, verbose=True, dtype='float32')

ratings = np.dot(U, V.T)
print(np.min(ratings))

pre5 = 0.0
pre10 = 0.0
pre20 = 0.0
rec5 = 0.0
rec10 = 0.0
rec20 = 0.0
mrr = 0.0

n_user = len(taboo)

for i in range(n_user):
	scores = ratings[i, :].copy()
	scores[taboo[i]] = -100
	res = np.argsort(-scores)
	u_mrr = 0.0

	res5 = res[0:5]
	res10 = res[0:10]
	res20 = res[0:20]

	pre5 += 1.0 * len(set(res5) & set(gr[i])) / 5
	rec5 += 1.0 * len(set(res5) & set(gr[i])) / len(gr[i])
	pre10 += 1.0 * len(set(res10) & set(gr[i])) / 10
	rec10 += 1.0 * len(set(res10) & set(gr[i])) / len(gr[i])
	pre20 += 1.0 * len(set(res20) & set(gr[i])) / 20
	rec20 += 1.0 * len(set(res20) & set(gr[i])) / len(gr[i])
	for mid in gr[i]:
		u_mrr += 1.0 / (np.argwhere(res == mid)[0, 0] + 1)
	mrr += 1.0*u_mrr / len(gr[i])


print("Mean Precision: " + str(pre5 / n_user) + "\t" + str(pre10 / n_user) + "\t" + str(pre20 / n_user))
print("Mean Recall   : " + str(rec5 / n_user) + "\t" + str(rec10 / n_user) + "\t" + str(rec20 / n_user))
print("MRR: ", mrr/n_user)

