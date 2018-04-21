# -*- coding: utf-8 -*-

import pickle
import numpy as np
import random


class NetSampler:
    def next_batch(self):
        n_case = self.samples.shape[0]
        np.random.shuffle(self.samples)

        negs = np.random.randint(0, self.n_poi, (n_case, self.n_neg))

        records = []
        labels = []
        for i in range(n_case):
            uid = self.samples[i, 0]
            pid = self.samples[i, 1]
            pre_pid = self.samples[i, 2]
            tid = self.samples[i, 3]

            records.append([uid, pid, pre_pid, tid])
            labels.append(1)
            for j in range(self.n_neg):
                while negs[i, j] == pid:
                    negs[i, j] = random.randint(0, self.n_poi - 1)
                records.append([uid, negs[i, j], pre_pid, tid])
                labels.append(0)
        records = np.asarray(records)
        labels = np.asarray(labels)

        n_samples = len(labels)

        batch_size = int(n_samples / self.n_batch)
        n_batch = int(n_samples / batch_size) if n_samples % batch_size == 0 else int(n_samples / batch_size) + 1

        for i in range(n_batch):
            yield records[i * batch_size: (i + 1) * batch_size, :], labels[i * batch_size: (i + 1) * batch_size]

    def next_batch_bpr(self):
        n_case = self.samples.shape[0]
        np.random.shuffle(self.samples)

        negs = np.random.randint(0, self.n_poi, (n_case,))

        records = []
        for i in range(n_case):
            uid = self.samples[i, 0]
            pid = self.samples[i, 1]
            pre_pid = self.samples[i, 2]
            tid = self.samples[i, 3]

            while negs[i] == pid:
                negs[i] = random.randint(0, self.n_poi - 1)
            records.append([uid, pid, pre_pid, tid])
            records.append([uid, negs[i], pre_pid, tid])
        records = np.asarray(records)

        n_samples = n_case

        batch_size = int(n_samples / self.n_batch)
        n_batch = int(n_samples / batch_size) if n_samples % batch_size == 0 else int(n_samples / batch_size) + 1

        for i in range(n_batch):
            yield records[i * 2 * batch_size: (i + 1) * 2 * batch_size, :]

    def next_batch_hard(self):
        n_case = self.samples.shape[0]
        np.random.shuffle(self.samples)

        negs = np.random.randint(0, self.n_poi, (n_case, self.n_neg))

        records = []
        for i in range(n_case):
            uid = self.samples[i, 0]
            pid = self.samples[i, 1]
            pre_pid = self.samples[i, 2]
            tid = self.samples[i, 3]

            records.append([uid, pid, pre_pid, tid])
            for j in range(self.n_neg):
                while negs[i, j] == pid:
                    negs[i, j] = random.randint(0, self.n_poi - 1)
                records.append([uid, negs[i, j], pre_pid, tid])
        records = np.asarray(records)

        n_samples = n_case

        batch_size = int(n_samples / self.n_batch)
        n_batch = int(n_samples / batch_size) if n_samples % batch_size == 0 else int(n_samples / batch_size) + 1

        for i in range(n_batch):
            batch_samples = records[i * (self.n_neg + 1) * batch_size: (i + 1) * (self.n_neg + 1) * batch_size, :]
            ac_batch_size = int(batch_samples.shape[0] / (self.n_neg + 1))
            yield batch_samples, np.asarray([1] * ac_batch_size + [0] * ac_batch_size)

    def __init__(self, dataset, n_neg=1, n_batch=50):
        print("Initialize Dataset: ", dataset)

        with open("../" + dataset + "/vtr.pkl", "rb") as f:
            [vi, vtr, vte] = pickle.load(f, encoding="utf8")

        with open("../" + dataset + "/maps.pkl", "rb") as f:
            [user_id, poi_id, cat_id, id_user, id_poi, id_cat] = pickle.load(f, encoding="utf8")

        with open(dataset + "/co.pkl", "rb") as f:
            [u_co, s_co, t_co, v_co] = pickle.load(f, encoding="utf8")

        with open(dataset + "/geo.pkl", "rb") as f:
            [u_geo, s_geo, t_geo, v_geo] = pickle.load(f, encoding="utf8")

        with open(dataset + "/cat.pkl", "rb") as f:
            [u_cat, s_cat, t_cat, v_cat] = pickle.load(f, encoding="utf8")

        with open(dataset + "/time.pkl", "rb") as f:
            [u_tim, query_time, v_tim] = pickle.load(f, encoding="utf8")

        with open(dataset + "/train.pkl", "rb") as f:
            raw_samples = pickle.load(f, encoding="utf8")

        with open(dataset + "/test.pkl", "rb") as f:
            raw_cases = pickle.load(f, encoding="utf8")

        samples = []
        cases = []
        taboo = {}
        gr = {}

        for record in raw_samples:
            samples.append([record[0], record[1], record[2], record[3]])

        for record in raw_cases:
            cases.append([record[0], record[1], record[2], record[3]])

        for uid in vtr:
            taboo[uid] = set(vtr[uid])
            gr[uid] = set(vte[uid])

        self.n_user = len(user_id)
        self.n_poi = len(poi_id)
        self.n_cat = len(cat_id)
        self.n_time = 48

        self.samples = np.asarray(samples, dtype=int)
        self.cases = np.asarray(cases, dtype=int)
        self.n_neg = n_neg
        self.vtr = vtr
        self.taboo = taboo
        self.n_batch = n_batch

        self.u_co = u_co
        self.s_co = s_co
        self.t_co = t_co
        self.v_co = v_co

        self.u_geo = u_geo
        self.s_geo = s_geo
        self.t_geo = t_geo
        self.v_geo = v_geo

        self.u_cat = u_cat
        self.s_cat = s_cat
        self.t_cat = t_cat
        self.v_cat = v_cat

        self.u_tim = u_tim
        self.v_tim = v_tim
        self.query_time = query_time

# tsp = NetSampler(dataset="fs", n_neg=20, n_batch=100)
# for sa, la in tsp.next_batch_hard():
# 	print(sa.shape, la.shape)
