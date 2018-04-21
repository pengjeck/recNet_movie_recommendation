# -*- coding: utf-8 -*-

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
        # network = tl.layers.DropoutLayer(network, keep=0.7, name='drop0')
        # network = tl.layers.DenseLayer(network, 1024, name='relu0')
        # network = tl.layers.BatchNormLayer(network, act=tf.nn.relu, name="bn0", is_train=is_train)

        network = tl.layers.DropoutLayer(network, keep=0.7, name='drop1')
        network = tl.layers.DenseLayer(network, 512, name='relu2')
        network = tl.layers.BatchNormLayer(network, act=tf.nn.relu, name="bn1", is_train=is_train)

        network = tl.layers.DropoutLayer(network, keep=0.7, name='drop2')
        network = tl.layers.DenseLayer(network, 256, name='relu3')
        network = tl.layers.BatchNormLayer(network, act=tf.nn.relu, name="bn2", is_train=is_train)

        network = tl.layers.DropoutLayer(network, keep=0.7, name='drop3')
        network = tl.layers.DenseLayer(network, n_units=1, act=tf.identity, name='output')

    return network


def bin_loss(samples, labels, lamb, u_co, v_co, u_geo, v_geo, u_cat, v_cat, s_co, s_geo, s_cat, t_co, t_geo, t_cat,
             n_neg):
    vec_u = tf.Variable(u_co, dtype=tf.float32)
    vec_s = tf.Variable(s_co, dtype=tf.float32)
    vec_t = tf.Variable(t_co, dtype=tf.float32)
    vec_v = tf.Variable(v_co, dtype=tf.float32)

    vec_u_geo = tf.Variable(u_geo, dtype=tf.float32)
    vec_s_geo = tf.Variable(s_geo, dtype=tf.float32)
    vec_t_geo = tf.Variable(t_geo, dtype=tf.float32)
    vec_v_geo = tf.Variable(v_geo, dtype=tf.float32)

    vec_u_cat = tf.Variable(u_cat, dtype=tf.float32)
    vec_s_cat = tf.Variable(s_cat, dtype=tf.float32)
    vec_t_cat = tf.Variable(t_cat, dtype=tf.float32)
    vec_v_cat = tf.Variable(v_cat, dtype=tf.float32)

    u = tf.nn.embedding_lookup(vec_u, samples[:, 0], name="users")
    u_g = tf.nn.embedding_lookup(vec_u_geo, samples[:, 0], name="users_geo")
    u_ca = tf.nn.embedding_lookup(vec_u_cat, samples[:, 0], name="users_cat")

    s = tf.nn.embedding_lookup(vec_s, samples[:, 2], name="seq")
    s_g = tf.nn.embedding_lookup(vec_s_geo, samples[:, 2], name="seq_geo")
    s_ca = tf.nn.embedding_lookup(vec_s_cat, samples[:, 2], name="seq_cat")

    v = tf.nn.embedding_lookup(vec_v, samples[:, 1], name="po_items")
    v_g = tf.nn.embedding_lookup(vec_v_geo, samples[:, 1], name="po_items_geo")
    v_ca = tf.nn.embedding_lookup(vec_v_cat, samples[:, 1], name="po_items_cat")

    pre_v = tf.nn.embedding_lookup(vec_v, samples[:, 2], name="pre_items")
    pre_v_g = tf.nn.embedding_lookup(vec_v_geo, samples[:, 2], name="pre_items_geo")
    pre_v_ca = tf.nn.embedding_lookup(vec_v_cat, samples[:, 2], name="pre_items_cat")

    t = tf.nn.embedding_lookup(vec_t, samples[:, 3], name="times")
    t_g = tf.nn.embedding_lookup(vec_t_geo, samples[:, 3], name="times_geo")
    t_ca = tf.nn.embedding_lookup(vec_t_cat, samples[:, 3], name="times_cat")

    vec = tf.concat([u, u_g, u_ca, t, t_g, t_ca, s, s_g, s_ca, pre_v, pre_v_g, pre_v_ca, v, v_g, v_ca], axis=1)

    network = network_predict(vec, is_train=True, reuse=False)
    network_test = network_predict(vec, is_train=False, reuse=True)

    predict = network.outputs[:, 0]
    predict_test = network_test.outputs[:, 0]

    scores = tf.reshape(predict, [-1, 1 + n_neg])
    pos_scores = scores[:, 0]
    neg_scores = tf.reduce_max(scores[:, 1:], axis=1)

    loss = tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.concat([pos_scores, neg_scores], axis=0), labels=labels))

    regularization_term = 0
    for para in network.all_params:
        regularization_term += l2_regularizer(lamb)(para)

    loss += regularization_term
    # para = [vec_u, vec_v, user_geo, loc_geo]
    # para.extend(network.all_params)
    para = network.all_params
    # para.extend([vec_u, vec_v, loc_geo, user_geo, user_cat, loc_cat, user_meta, loc_meta])

    return loss, network, network_test, para, predict_test


def train_rec_net(dataset, n_neg, n_epoch, lamb, lr, n_batch):
    sess = tf.InteractiveSession()

    sampler = NetSampler(dataset, n_neg=n_neg, n_batch=n_batch)
    samples = tf.placeholder(tf.int32, [None, 4], name="po")
    labels = tf.placeholder(tf.float32, [None, ], name="la")

    loss, network, network_test, para, predict = bin_loss(samples, labels, lamb, sampler.u_co, sampler.v_co,
                                                          sampler.u_geo, sampler.v_geo, sampler.u_cat, sampler.v_cat,
                                                          sampler.s_co, sampler.s_geo, sampler.s_cat, sampler.t_co,
                                                          sampler.t_geo, sampler.t_cat, sampler.n_neg)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=para)

    sess.run(tf.initialize_all_variables())
    network.print_params()

    log = open("result.log", "w", encoding="utf8")
    log.writelines("Dataset: " + dataset + "\n")

    for j in range(n_epoch):
        total_loss = 0
        st = time.time()
        for pos_samples, lb in sampler.next_batch_hard():
            feed_dict = {
                samples: pos_samples,
                labels: lb
            }
            feed_dict.update(network.all_drop)
            _, bat_loss = sess.run((optimizer, loss), feed_dict=feed_dict)
            total_loss += bat_loss

        print("Iteration ", j, " eclapsed ", time.time() - st, " seconds, loss is: ", total_loss)

        # if j % 4 == 0 and j > 30:
        if j % 4 == 0 and j > 100:
            acc5 = 0.0
            acc10 = 0.0
            acc20 = 0.0
            mrr = 0.0
            count = 0.0

            for record in sampler.cases:
                uid = record[0]
                pid = record[1]
                pre_pid = record[2]
                tid = record[3]

                dp_dict = tl.utils.dict_to_one(network_test.all_drop)
                feed_dict = {
                    samples: np.asarray([[uid] * sampler.n_movie, list(range(sampler.n_movie)), [pre_pid] * sampler.n_movie,
                                         [tid] * sampler.n_movie], dtype=int).T
                }
                feed_dict.update(dp_dict)
                ratings = sess.run(predict, feed_dict=feed_dict)
                ratings[sampler.vtr[uid]] = -1000
                res = np.argsort(-ratings)

                if pid in res[0:5]:
                    acc5 += 1
                    acc10 += 1
                    acc20 += 1
                elif pid in res[0:10]:
                    acc10 += 1
                    acc20 += 1
                elif pid in res[0:20]:
                    acc20 += 1

                mrr += 1 / (np.argwhere(res == pid)[0, 0] + 1)
                count += 1

                if count % 10000 == 0:
                    print(count)

            print("Acc@5: ", acc5 / count, "Acc@10: ", acc10 / count, "Acc@20: ", acc20 / count)
            print(mrr / count)
            print("")

            log.writelines("Iteration: " + str(j) + ", loss: " + str(total_loss) + "\n")
            log.writelines("Acc@5: " + str(acc5 / count) + ", Acc@10: " +
                           str(acc10 / count) + ", Acc@20: " + str(acc20 / count) + "\n")
            log.writelines("MRR: " + str(mrr / count) + "\n")
            log.writelines("\n")

            log.flush()

    log.close()


def rec_net(dataset, n_neg=20, n_epoch=400, lamb=0.00001, lr=0.0003, n_batch=100):
    print("learning rate", lr)
    print("regularization term", lamb)

    train_rec_net(dataset, n_neg, n_epoch, lamb, lr, n_batch)


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
rec_net("fs")
