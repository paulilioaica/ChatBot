import json
from functools import reduce

import numpy as np

char_list = ['?','!()']

def load_train_data(path):
    with open(path, 'r') as f:
        return json.load(f)



def get_vocab(data):
    all_words = set([reduce(lambda x,y : x.replace(y,"") ,[',', '!', '.', ';',"'",'"',"/",   '?', '!', '(', ")"], word).lower() for unpacked in data for sentance in unpacked for word in sentance.split()])
    vocab = {word: i for i,word in enumerate(all_words)}
    return vocab

def get_training_data(batch_size=1):
    training_data = []
    training_dict = load_train_data('data/train.json')
    for key in training_dict:
        training_data.append(training_dict[key])
    vocab = get_vocab(training_data)
    print(vocab)
    training_count = len(training_data)
    if training_count % batch_size != 0:
        print("Warning, batch size does not divide total training data exactly")
    total_batches = training_count // batch_size
    batches = np.array((total_batches, batch_size))
    for i in range(0, total_batches):
        for j in range(batch_size):
            batches[i][j] = np.array([vocab[word] for word in training_data[i*batch_size:(i*batch_size + batch_size)]])
    print(batches.shape)


get_training_data(batch_size=4)