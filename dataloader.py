import json
from functools import reduce

import numpy as np

char_list = ['?', '!()']


def load_train_data(path):
    with open(path, 'r') as f:
        return json.load(f)


def strip_word(word):
    return reduce(lambda x, y: x.replace(y, ""), [',', '!', '.', ';', "'", '"', "/", '?', '!', '(', ")"], word).lower()


def get_vocab(data):
    all_words = set([strip_word(word) for unpacked in data for sentance in unpacked for word in sentance.split()])
    vocab = {word: i for i, word in enumerate(all_words)}
    return vocab


def get_training_data(batch_size=1):
    training_data = []
    training_dict = load_train_data('data/train.json')
    for key in training_dict:
        training_data.append(training_dict[key])

    vocab = get_vocab(training_data)
    max_len_question = 43
    print(len(vocab))


    for i in range(len(training_data)):
        training_data[i][0] = [vocab[word] for word in strip_word(training_data[i][0]).split()]
        training_data[i][1] = [vocab[word] for word in strip_word(training_data[i][1]).split()]

    training_count = len(training_data)
    if training_count % batch_size != 0:
        print("Warning, batch size does not divide total training data exactly")
    total_batches = training_count // batch_size
    X_train = np.zeros((total_batches, batch_size, max_len_question))
    Y_train = np.zeros((total_batches, batch_size, max_len_question))
    for i in range(0, total_batches):
        q = [qna[0] + [0]*(max_len_question-len(qna[0])) for qna in training_data[i * batch_size:(i * batch_size + batch_size)]]
        a = [qna[1] + [0]*(max_len_question-len(qna[1])) for qna in training_data[i * batch_size:(i * batch_size + batch_size)]]

        for number in range(batch_size):
            X_train[i][number] = q[number]
        for number in range(batch_size):
            Y_train[i][number] = a[number]
    return X_train, Y_train

get_training_data()