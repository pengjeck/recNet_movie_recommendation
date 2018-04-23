# -*- coding:utf-8 -*-

import pickle
import numpy as np
import random


class BPRSampler:
    def next_batch(self):
        n_case = self.samples.shape[0]
        np.random.shuffle(self.samples)

        negs = np.random.randint(0, self.n_movie, (n_case,))

        records = []
        for i in range(n_case):
            uid = self.samples[i, 0]
            mid = self.samples[i, 1]
            while negs[i] in self.taboo[uid]:
                negs[i] = random.randint(0, self.n_movie - 1)
            records.append([uid, mid, negs[i]])
        records = np.asarray(records)

        batch_size = int(n_case / self.n_batch)
        n_batch = int(n_case / batch_size) if n_case % batch_size == 0 else int(n_case / batch_size) + 1

        for i in range(n_batch):
            yield records[i * batch_size: (i + 1) * batch_size, :]

    def __init__(self, dataset, n_batch=50):
        with open("../data/" + dataset + "/data.pkl", "rb") as f:
            [mtr, mte] = pickle.load(f, encoding="utf8")

        with open("../data/" + dataset + "/maps.pkl", "rb") as f:
            [user_id, id_user, movie_id, id_movie] = pickle.load(f, encoding="utf8")

        samples = []
        taboo = {}
        for uid in mtr:
            taboo[uid] = set(mtr[uid])
            for mid in mtr[uid]:
                samples.append([uid, mid])
        print(len(samples))

        self.n_user = len(user_id)
        self.n_movie = len(movie_id)
        self.gr = mte
        self.taboo = taboo
        self.samples = np.asarray(samples, dtype=int)
        self.n_batch = n_batch

#
# sp = BPRSampler("m1m")
# index = 1
# for sa in sp.next_batch():
# 	print(index, sa.shape[0])
# 	index += 1
