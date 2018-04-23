# -*- coding: utf-8 -*-

import pickle
import numpy as np
from sklearn.decomposition import NMF
from config import *
from time import time, sleep

co_matrix = np.zeros((n_movie, n_movie), dtype=float)


def create_vtr():
    """
    训练集
    vtr = {
        'uid': ['pid1', 'pid2']
    }
    :return:
    """
    res = {}
    with open(new_rating_file_path, 'r') as f:
        while True:
            line = f.readline()
            if len(line) <= 0:
                break

            parts = line.split(',')
            user_id = int(parts[0])
            if user_id not in res:
                res[user_id] = [int(parts[1])]
            else:
                res[user_id].append(int(parts[1]))
    return res


def create_vte():
    """
    构建测试集
    :return:
    """
    return {}


# vtr = create_vtr()
# vte = create_vte()

# with open(new_base_path + 'vtr.pkl', 'wb') as f:
#     pickle.dump([vtr], f)
#     f.flush()
# sleep(10)
# exit(0)

with open(new_base_path + 'vtr.pkl', 'rb') as f:
    [vtr] = pickle.load(f)

beg_t = time()

# 这样会自动的遍历 dict的key
for uid in vtr:
    # 当前用户访问过多少
    count = len(vtr[uid])

    for i in range(count):
        pid1 = vtr[uid][i]
        co_matrix[pid1, pid1] += 1
        for j in range(i + 1, count):
            pid2 = vtr[uid][j]
            co_matrix[pid1, pid2] += 1
            co_matrix[pid2, pid1] += 1

t1 = time()
print(t1 - beg_t)

# 分解得到：(n * 100) * (100 * m)
model = NMF(n_components=100, init='nndsvd')
model.fit(co_matrix)
H = model.components_

t2 = time()
print(t2 - t1)

# Vec.shape = (m, 100)
Vec = H.T
u_g = np.zeros((n_user, 100), dtype=float)

# 遍每一个用户
for i in range(n_user):
    # 得到每个用户访问的地点
    for pid in vtr[i]:
        u_g[i] += Vec[pid]
    u_g[i] /= len(vtr[i])
# u_g[i]代表均值

t3 = time()
print(t3 - t2)

mean_v = np.mean(Vec)
Vec_n = Vec - mean_v
print(mean_v)

mean_u = np.mean(u_g)
u = u_g - mean_u
print(mean_u)

with open(new_base_path + "/co.pkl", "wb") as f:
    pickle.dump([u_g, Vec], f)
