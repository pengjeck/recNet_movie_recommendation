# -*- coding:utf-8 -*-

import tensorflow as tf
from RecSampler import BPRSampler

import time
import numpy as np
import os


def bpr_loss(samples, n_user, n_movie, n_dim, lamb, test_user):
    vec_u = tf.Variable(tf.random_normal([n_user, n_dim], stddev=1 / n_dim, dtype=tf.float32))
    vec_v = tf.Variable(tf.random_normal([n_movie, n_dim], stddev=1 / n_dim, dtype=tf.float32))

    u = tf.nn.embedding_lookup(vec_u, samples[:, 0], name="users")
    pv = tf.nn.embedding_lookup(vec_v, samples[:, 1], name="pos")
    nv = tf.nn.embedding_lookup(vec_v, samples[:, 2], name="neg")

    pos_score = tf.reduce_sum(u * pv, 1)
    neg_score = tf.reduce_sum(u * nv, 1)

    loss = -1 * tf.reduce_sum(tf.log(tf.sigmoid(pos_score - neg_score)))
    regularization_term = lamb * (tf.reduce_sum(tf.square(vec_u)) + tf.reduce_sum(tf.square(vec_v)))
    loss += regularization_term

    test_u = tf.nn.embedding_lookup(vec_u, test_user, name="test_uid")
    predict = tf.reduce_sum(test_u * vec_v, 1)

    return loss, predict, vec_u, vec_v


def train_bpr(n_dim, n_epoch, lamb, lr):
    sess = tf.InteractiveSession()

    sampler = BPRSampler()
    samples = tf.placeholder(tf.int32, [None, 3], name="samp")
    test_user = tf.placeholder(tf.int32, [1], name="test_user")

    print(sampler.n_user, sampler.n_movie)

    loss, predict, vec_u, vec_v = bpr_loss(samples, sampler.n_user, sampler.n_movie, n_dim, lamb, test_user)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=[vec_u, vec_v])

    sess.run(tf.global_variables_initializer())

    for j in range(n_epoch):
        total_loss = 0
        st = time.time()
        for pos_samples in sampler.next_batch():
            feed_dict = {
                samples: pos_samples,
            }
            _, bat_loss = sess.run((optimizer, loss), feed_dict=feed_dict)
            total_loss += bat_loss
        print("Iteration ", j, " eclapsed ", time.time() - st, " seconds, loss is: ", total_loss)

        if j % 2 == 0 and j > 100:
            pre5 = 0.0
            pre10 = 0.0
            pre20 = 0.0
            rec5 = 0.0
            rec10 = 0.0
            rec20 = 0.0
            mrr = 0
            n_user_t = sampler.n_user

            for uid in range(sampler.n_user):
                feed_dict = {
                    test_user: [uid]
                }
                ratings = sess.run(predict, feed_dict=feed_dict)
                ratings[list(sampler.taboo[uid])] = -1000
                res = np.argsort(-ratings)
                u_mrr = 0

                res5 = res[0:5]
                res10 = res[0:10]
                res20 = res[0:20]

                pre5 += 1.0 * len(set(res5) & set(sampler.mte[uid])) / 5
                rec5 += 1.0 * len(set(res5) & set(sampler.mte[uid])) / len(sampler.mte[uid])
                pre10 += 1.0 * len(set(res10) & set(sampler.mte[uid])) / 10
                rec10 += 1.0 * len(set(res10) & set(sampler.mte[uid])) / len(sampler.mte[uid])
                pre20 += 1.0 * len(set(res20) & set(sampler.mte[uid])) / 20
                rec20 += 1.0 * len(set(res20) & set(sampler.mte[uid])) / len(sampler.mte[uid])

                if len(sampler.mte[uid]):
                    for mid in sampler.mte[uid]:
                        u_mrr += 1 / (np.argwhere(res == mid)[0, 0] + 1)
                    mrr += u_mrr / len(sampler.mte[uid])
                else:
                    n_user_t -= 1

            print("Mean Precision: " + str(pre5 / n_user_t) + "\t" + str(pre10 / n_user_t) + "\t" + str(
                pre20 / n_user_t))
            print("Mean Recall   : " + str(rec5 / n_user_t) + "\t" + str(rec10 / n_user_t) + "\t" + str(
                rec20 / n_user_t))
            print("MRR: ", mrr / n_user_t)
            print("")


def bpr_mf(n_dim=100, n_epoch=1000, lamb=0.0001, lr=0.0005):
    train_bpr(n_dim, n_epoch, lamb, lr)


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
bpr_mf()
