# coding:utf-8
from config import *
import pandas as pd
import os


def movie2index():
    movie_ids = {}
    with open(movie_file_path, 'r') as f:
        counter = 0
        while True:
            line = f.readline()
            if len(line) <= 0:
                break
            parts = line.split(',')
            movie_ids[int(parts[0])] = counter
            counter += 1
    print(len(movie_ids), n_movie)
    return movie_ids


def convert_rating_movie_id():
    with open(rating_file_path, 'r') as rating_f, \
            open(new_base_path + 'ratings.csv', 'w') as new_rating_f:
        movie_ids = movie2index()
        while True:
            line = rating_f.readline()
            if len(line) <= 0:
                break
            parts = line.split(',')
            # 只有评分大于或等于3的才会被记为正样本
            if float(parts[2]) >= 3:
                new_rating_f.write('{},{}\n'.format(int(parts[0]) - 1, movie_ids[int(parts[1])]))


encoding = 'latin1'

u_path = os.path.expanduser('ch02/ml-1m/users.dat')
r_path = os.path.expanduser('data/ml-1m/ratings.dat')
m_path = os.path.expanduser('ch02/ml-1m/movies.dat')

u_names = ['user_id', 'gender', 'age', 'occupation', 'zip']
r_names = ['user_id', 'movie_id', 'rating', 'timestamp']
m_names = ['movie_id', 'title', 'genres']

users = pd.read_csv(u_path, sep='::', header=None, names=u_names, encoding=encoding)
ratings = pd.read_csv(r_path, sep='::', header=None, names=r_names, encoding=encoding)
movies = pd.read_csv(m_path, sep='::', header=None, names=m_names, encoding=encoding)

