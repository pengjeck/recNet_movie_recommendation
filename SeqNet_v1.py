# -*- coding: utf-8 -*-
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl
from tensorflow.contrib.layers import l2_regularizer
from sampler import NetSampler
import time
import numpy as np
import os


def network_predict(x, is_train=True, reuse=False):
    with tf.variable_scope("rec_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network = tl.layers.InputLayer(x, name='input')

        # network = tl.layers.DropoutLayer(network, keep=0.7, name='drop5')
        # network = tl.layers.DenseLayer(network, 2048, name='relu5')
        # network = tl.layers.BatchNormLayer(network, act=tf.nn.relu, name="bn5", is_train=is_train)
        #
        network = tl.layers.DropoutLayer(network, keep=0.7, name='drop0')
        network = tl.layers.DenseLayer(network, 1024, name='relu0')
        network = tl.layers.BatchNormLayer(network, act=tf.nn.relu, name="bn0", is_train=is_train)

        network = tl.layers.DropoutLayer(network, keep=0.7, name='drop1')
        network = tl.layers.DenseLayer(network, 512, name='relu2')
        network = tl.layers.BatchNormLayer(network, act=tf.nn.relu, name="bn1", is_train=is_train)

        network = tl.layers.DropoutLayer(network, keep=0.7, name='drop2')
        network = tl.layers.DenseLayer(network, 256, name='relu3')
        network = tl.layers.BatchNormLayer(network, act=tf.nn.relu, name="bn2", is_train=is_train)

        network = tl.layers.DropoutLayer(network, keep=0.7, name='drop3')
        network = tl.layers.DenseLayer(network, n_units=1, act=tf.identity, name='output')

    return network


def bin_loss(samples, labels, lamb, u_co, v_co, n_neg):
    vec_u = tf.Variable(u_co, dtype=tf.float32)
    vec_v = tf.Variable(v_co, dtype=tf.float32)

    u = tf.nn.embedding_lookup(vec_u, samples[:, 0], name="users")

    v = tf.nn.embedding_lookup(vec_v, samples[:, 1], name="po_items")

    # u v 连接在一起作为一个向量
    vec = tf.concat([u, v], axis=1)

    network = network_predict(vec, is_train=True, reuse=False)
    network_test = network_predict(vec, is_train=False, reuse=True)

    predict = network.outputs[:, 0]
    predict_test = network_test.outputs[:, 0]

    scores = tf.reshape(predict, [-1, 1 + n_neg])
    pos_scores = scores[:, 0]
    # 计算最大误差的的得分
    neg_scores = tf.reduce_max(scores[:, 1:], axis=1)

    loss = tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.concat([pos_scores, neg_scores], axis=0),
                                                labels=labels))

    regularization_term = 0
    for para in network.all_params:
        regularization_term += l2_regularizer(lamb)(para)

    loss += regularization_term
    # para = [vec_u, vec_v, user_geo, loc_geo]
    # para.extend(network.all_params)
    para = network.all_params
    # para.extend([vec_u, vec_v, loc_geo, user_geo, user_cat, loc_cat, user_meta, loc_meta])

    return loss, network, network_test, para, predict_test


def train_rec_net(n_neg, n_epoch, lamb, lr, n_batch):
    sess = tf.InteractiveSession()

    sampler = NetSampler(n_neg=n_neg, n_batch=n_batch)
    # 样本和标签的placeholder
    samples = tf.placeholder(tf.int32, [None, 2], name="po")
    labels = tf.placeholder(tf.float32, [None, ], name="la")

    loss, network, network_test, para, predict = bin_loss(samples,
                                                          labels,
                                                          lamb,
                                                          sampler.u_co,
                                                          sampler.v_co,
                                                          sampler.n_neg)
    # 优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=para)

    # 运行网络？
    sess.run(tf.global_variables_initializer())
    network.print_params()

    log = open("result.log", "w", encoding="utf8")

    for j in range(n_epoch):
        total_loss = 0
        st = time.time()
        # 训练网络
        for pos_samples, lb in sampler.next_batch_hard():
            # for pos_samples, lb in sampler.next_batch():
            feed_dict = {
                samples: pos_samples,
                labels: lb
            }
            feed_dict.update(network.all_drop)
            _, bat_loss = sess.run((optimizer, loss),
                                   feed_dict=feed_dict)
            total_loss += bat_loss

        # 第几轮，耗时多少，损失多少，
        print("Iteration ", j, " eclapsed ", time.time() - st, " seconds, loss is: ", total_loss)

        # 计算相关的评价参数的值，这里我们要使用准确率和召回率，需要自己去写
        # 每4轮计算一次准确率和召回率，前面300轮不计算
        if j % 4 == 0 and j > 300:
            pre5 = 0.0
            pre10 = 0.0
            pre20 = 0.0
            rec5 = 0.0
            rec10 = 0.0
            rec20 = 0.0
            mrr = 0
            n_user_t = sampler.n_user

            for u_id in range(sampler.n_user):
                dp_dict = tl.utils.dict_to_one(network_test.all_drop)
                feed_dict = {
                    samples: np.asarray([[u_id] * sampler.n_movie,
                                         list(range(sampler.n_movie))], dtype=int).T
                }
                feed_dict.update(dp_dict)
                ratings = sess.run(predict, feed_dict=feed_dict)
                ratings[list(sampler.taboo[u_id])] = -1000
                res = np.argsort(-ratings)

                u_mrr = 0

                res5 = res[0:5]
                res10 = res[0:10]
                res20 = res[0:20]

                pre5 += 1.0 * len(set(res5) & set(sampler.mte[u_id])) / 5
                rec5 += 1.0 * len(set(res5) & set(sampler.mte[u_id])) / len(sampler.mte[u_id])
                pre10 += 1.0 * len(set(res10) & set(sampler.mte[u_id])) / 10
                rec10 += 1.0 * len(set(res10) & set(sampler.mte[u_id])) / len(sampler.mte[u_id])
                pre20 += 1.0 * len(set(res20) & set(sampler.mte[u_id])) / 20
                rec20 += 1.0 * len(set(res20) & set(sampler.mte[u_id])) / len(sampler.mte[u_id])

                if len(sampler.mte[u_id]):
                    for mid in sampler.mte[u_id]:
                        u_mrr += 1 / (np.argwhere(res == mid)[0, 0] + 1)
                    mrr += u_mrr / len(sampler.mte[u_id])
                else:
                    n_user_t -= 1

            print("Mean Precision: " + str(pre5 / n_user_t) + "\t" + str(pre10 / n_user_t) + "\t" + str(
                pre20 / n_user_t))
            print("Mean Recall   : " + str(rec5 / n_user_t) + "\t" + str(rec10 / n_user_t) + "\t" + str(
                rec20 / n_user_t))
            print("MRR: ", mrr / n_user_t)
            print("")

            log.writelines("Mean Precision: " + str(pre5 / n_user_t) + "\t" + str(pre10 / n_user_t) + "\t" + str(
                pre20 / n_user_t))
            log.writelines("Mean Recall   : " + str(rec5 / n_user_t) + "\t" + str(rec10 / n_user_t) + "\t" + str(
                rec20 / n_user_t))
            log.writelines("MRR: " + str(mrr / n_user_t))
            log.writelines("\n")

            log.flush()
    log.close()


def rec_net(n_neg=4, n_epoch=400, lamb=0.0001, lr=0.0003, n_batch=100):
    print("learning rate", lr)
    print("regularization term", lamb)

    train_rec_net(n_neg, n_epoch, lamb, lr, n_batch)


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
rec_net()
