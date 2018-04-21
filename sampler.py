# -*- coding: utf-8 -*-

import pickle
import numpy as np
import random
from config import *


class NetSampler:
    def next_batch(self):
        """
        1. 每一次计算下一次的结果
        2. 每一个正样本生成self.n_neg个负样本
        3. 系统默认只给出正样本和以及正样本的个数
        :return:
        """
        n_case = self.samples.shape[0]
        np.random.shuffle(self.samples)

        negs = np.random.randint(0, self.n_movie, (n_case, self.n_neg))

        records = []
        labels = []
        for i in range(n_case):
            u_id = self.samples[i, 0]
            m_id = self.samples[i, 1]
            pre_pid = self.samples[i, 2]
            tid = self.samples[i, 3]

            records.append([u_id, m_id, pre_pid, tid])
            labels.append(1)
            for j in range(self.n_neg):
                # 当前列对应的movie不能是用户已经浏览过的，如果是的话，一定要重新随机生成
                while negs[i, j] == m_id:
                    # 负样本
                    negs[i, j] = random.randint(0, self.n_movie - 1)
                # 每一个正样本下面对应添加 self.n_neg 个负样本
                records.append([u_id, negs[i, j], pre_pid, tid])
                labels.append(0)
        records = np.asarray(records)
        labels = np.asarray(labels)

        # 总的样本数目
        n_samples = len(labels)

        # 总的样本数目除以分为多少batch， 就可以得到每个batch相应的大小
        batch_size = int(n_samples / self.n_batch)
        # 重新计算batch个数
        n_batch = int(n_samples / batch_size) if n_samples % batch_size == 0 else int(n_samples / batch_size) + 1

        for i in range(n_batch):
            yield records[i * batch_size: (i + 1) * batch_size, :], \
                  labels[i * batch_size: (i + 1) * batch_size]

    def next_batch_bpr(self):
        """
        1、每一正样本对应输出一个负样本
        2、不输出正负样本对应的label
        :return:
        """
        n_case = self.samples.shape[0]
        np.random.shuffle(self.samples)

        negs = np.random.randint(0, self.n_movie, (n_case,))

        records = []
        for i in range(n_case):
            uid = self.samples[i, 0]
            pid = self.samples[i, 1]
            pre_pid = self.samples[i, 2]
            tid = self.samples[i, 3]

            # 每个正样本对应只会生成一个负样本
            while negs[i] == pid:
                negs[i] = random.randint(0, self.n_movie - 1)
            records.append([uid, pid, pre_pid, tid])
            records.append([uid, negs[i], pre_pid, tid])
        records = np.asarray(records)

        # 这里n_samples是正样本的个数
        n_samples = n_case

        batch_size = int(n_samples / self.n_batch)
        n_batch = int(n_samples / batch_size) if n_samples % batch_size == 0 else int(n_samples / batch_size) + 1

        for i in range(n_batch):
            # 因为batch_size是对应正样本的个数，而且一个正样本对应一个负样本，所以这里乘以2
            yield records[i * 2 * batch_size: (i + 1) * 2 * batch_size, :]

    def next_batch_hard(self):
        """
        1. 一个正样本对应self.n_neg个负样本
        2. 输出样本情况，以及对应的label
        这一段代码有问题吧
        :return:
        """
        n_case = self.samples.shape[0]
        np.random.shuffle(self.samples)

        negs = np.random.randint(0, self.n_movie, (n_case, self.n_neg))

        records = []
        # 每一生成对应的标签，也就是隐式反馈的结果
        for i in range(n_case):
            uid = self.samples[i, 0]
            pid = self.samples[i, 1]
            pre_pid = self.samples[i, 2]
            tid = self.samples[i, 3]

            records.append([uid, pid, pre_pid, tid])
            for j in range(self.n_neg):
                while negs[i, j] == pid:
                    negs[i, j] = random.randint(0, self.n_movie - 1)
                records.append([uid, negs[i, j], pre_pid, tid])
        records = np.asarray(records)

        # n_samples 为正样本的个数
        n_samples = n_case

        # batch_size表示最后输出中正样本的个数
        batch_size = int(n_samples / self.n_batch)
        n_batch = int(n_samples / batch_size) if n_samples % batch_size == 0 else int(n_samples / batch_size) + 1

        for i in range(n_batch):
            # (self.n_net + 1) * batch_size 表示总的样本个数
            batch_samples = records[i * (self.n_neg + 1) * batch_size: (i + 1) * (self.n_neg + 1) * batch_size, :]
            # 这一批样本中，正样本有多少个
            ac_batch_size = int(batch_samples.shape[0] / (self.n_neg + 1))
            # 正负样本一样多？？
            yield batch_samples, np.asarray([1] * ac_batch_size + [0] * ac_batch_size)

    def __init__(self, dataset, n_neg=1, n_batch=50):
        print("Initialize Dataset: ", dataset)

        with open("../" + dataset + "/vtr.pkl", "rb") as f:
            [vi, vtr, vte] = pickle.load(f, encoding="utf8")

        # 只有两项
        with open(dataset + "/co.pkl", "rb") as f:
            [u_co, v_co] = pickle.load(f, encoding="utf8")

        # 训练数据
        with open(dataset + "/train.pkl", "rb") as f:
            raw_samples = pickle.load(f, encoding="utf8")

        # 测试数据
        with open(dataset + "/test.pkl", "rb") as f:
            raw_cases = pickle.load(f, encoding="utf8")

        samples = []
        cases = []

        for record in raw_samples:
            samples.append([record[0], record[1], record[2], record[3]])

        for record in raw_cases:
            cases.append([record[0], record[1], record[2], record[3]])

        self.vtr = vtr

        self.n_user = n_user
        self.n_movie = n_movie
        self.n_time = 48

        self.samples = np.asarray(samples, dtype=int)
        self.cases = np.asarray(cases, dtype=int)
        self.n_neg = n_neg
        self.n_batch = n_batch

        self.u_co = u_co
        self.v_co = v_co
