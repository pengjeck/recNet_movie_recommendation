# -*- coding: utf-8 -*-

import pickle
import numpy as np
from sklearn.decomposition import NMF

dataset = "fs"
print(dataset)

with open("../" + dataset + "/vtr.pkl", "rb") as f:
    [vi, vtr, vte] = pickle.load(f, encoding="utf8")

with open("../" + dataset + "/ctr.pkl", "rb") as f:
    [ci, ctr, cte] = pickle.load(f, encoding="utf8")

with open("../" + dataset + "/maps.pkl", "rb") as f:
    [user_id, poi_id, cat_id, id_user, id_poi, id_cat] = pickle.load(f, encoding="utf8")

with open(dataset + "/ts.pkl", "rb") as f:
    [si, ti] = pickle.load(f, encoding="utf8")

print(len(si))
print(len(ti))

n_poi = len(poi_id)
co_matrix = np.zeros((n_poi, n_poi), dtype=float)
for uid in vtr:
    count = len(vtr[uid])
    for i in range(count):
        pid1 = vtr[uid][i]
        co_matrix[pid1, pid1] += 1
        for j in range(i + 1, count):
            pid2 = vtr[uid][j]
            co_matrix[pid1, pid2] += 1
            co_matrix[pid2, pid1] += 1

model = NMF(n_components=100, init='nndsvd')
model.fit(co_matrix)
H = model.components_
Vec = H.T

u_g = np.zeros((len(user_id), 100), dtype=float)
for i in range(len(user_id)):
    for pid in vtr[i]:
        u_g[i] += Vec[pid]
    u_g[i] /= len(vtr[i])

s_s = np.zeros((n_poi, 100), dtype=float)
for i in range(n_poi):
    for pid in si[i]:
        s_s[i] += Vec[pid]
    if len(si[i]):
        s_s[i] /= len(si[i])

t_t = np.zeros((len(ti), 100), dtype=float)
for i in range(len(ti)):
    for pid in ti[i]:
        t_t[i] += Vec[pid]
    if len(ti[i]):
        t_t[i] /= len(ti[i])

mean_v = np.mean(Vec)
Vec_n = Vec - mean_v
print(mean_v)

mean_u = np.mean(u_g)
u = u_g - mean_u
print(mean_u)

mean_s = np.mean(s_s)
s = s_s - mean_s
print(mean_s)

mean_t = np.mean(t_t)
t = t_t - mean_t
print(mean_t)

with open(dataset + "/co.pkl", "wb") as f:
    pickle.dump([u_g, s_s, t_t, Vec], f)
