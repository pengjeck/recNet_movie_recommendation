# coding:utf-8
from config import *
import numpy as np
import pickle

raw_record = []
with open(new_rating_file_path, 'r') as f:
    while True:
        line = f.readline()
        if len(line) <= 0:
            break
        parts = [int(item.strip()) for item in line.split(',')]
        raw_record.append(parts)

records = np.asarray(raw_record, dtype=np.int32)
np.random.shuffle(records)
train_records = records[:int(0.7 * records.shape[0])]
test_records = records[int(0.7 * records.shape[0]):]
with open(new_base_path + 'train.pkl', 'wb') as f:
    pickle.dump(train_records, f)
    f.flush()

with open(new_base_path + 'test.pkl', 'wb') as f:
    pickle.dump(test_records, f)
    f.flush()
