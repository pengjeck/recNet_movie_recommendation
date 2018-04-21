# -*- coding: utf-8 -*-

# 地理位置相关矩阵

import pickle
import numpy as np
from sklearn.decomposition import NMF

dataset = "fs"
# 判断是否访问的阈值
thres = 8

print(dataset)
print(thres)

with open("../" + dataset + "/vtr.pkl", "rb") as f:
    [vi, vtr, vte] = pickle.load(f, encoding="utf8")

with open("../" + dataset + "/ctr.pkl", "rb") as f:
    [ci, ctr, cte] = pickle.load(f, encoding="utf8")

with open("../" + dataset + "/maps.pkl", "rb") as f:
    [user_id, poi_id, cat_id, id_user, id_poi, id_cat] = pickle.load(f, encoding="utf8")

# 距离矩阵
with open(dataset + "/dis.pkl", "rb") as f:
    dis = pickle.load(f, encoding="utf8")

with open(dataset + "/ts.pkl", "rb") as f:
    [si, ti] = pickle.load(f, encoding="utf8")

print(len(si))
print(len(ti))

# 有多少个poi点
n_poi = dis.shape[0]

# 初始矩阵
geo = np.zeros((n_poi, n_poi), dtype=float)
for i in range(n_poi):
    for j in range(i, n_poi):
        if dis[i, j] < thres:
            geo[i, j] = 1
            geo[j, i] = 1

# 矩阵分解模型，分解出100个features
model = NMF(n_components=100, init='nndsvd')
# 自动的利用已有的数据拟合
model.fit(geo)

# H矩阵
H = model.components_

# （n_components, features) => (features, n_components)
Vec = H.T

# (用户数，100）
u_g = np.zeros((len(user_id), 100), dtype=float)

# 每一个用户
for i in range(len(user_id)):
    # 每一个地点的id
    for pid in vtr[i]:
        u_g[i] += Vec[pid]
    u_g[i] /= len(vtr[i])

# 每一个poi
s_s = np.zeros((n_poi, 100), dtype=float)
for i in range(n_poi):
    for pid in si[i]:
        s_s[i] += Vec[pid]
    if len(si[i]):
        s_s[i] /= len(si[i])

#
t_t = np.zeros((len(ti), 100), dtype=float)
for i in range(len(ti)):
    for pid in ti[i]:
        t_t[i] += Vec[pid]
    if len(ti[i]):
        t_t[i] /= len(ti[i])

#
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

with open(dataset + "/geo.pkl", "wb") as f:
    pickle.dump([u_g, s_s, t_t, Vec], f)
