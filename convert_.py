# coding:utf-8

from config import *


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
            new_rating_f.write('{},{}\n'.format(int(parts[0]) - 1, movie_ids[int(parts[1])]))


def count_user():
    user_ids = []
    with open(rating_file_path, 'r') as f:
        while True:
            line = f.readline()
            if len(line) <= 0:
                break
            parts = line.split(',')
            user_ids.append(int(parts[0]))
    print(n_user, len(set(user_ids)))
convert_rating_movie_id()