import codecs
import numpy as np
import math
import random
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from keras.layers.core import Dense, Dropout
from kerastuner.tuners import RandomSearch
import kerastuner as kt

random.seed(10)


# def parameter_tuning()


def MLP(x_train, y_train, x_dev, y_dev, x_test, y_test):
    # Conver vectors to tensors
    x_train = tf.convert_to_tensor(x_train, dtype='float32')
    y_train = tf.convert_to_tensor(y_train, dtype='float32')
    x_dev = tf.convert_to_tensor(x_dev, dtype='float32')
    y_dev = tf.convert_to_tensor(y_dev, dtype='float32')
    x_test = tf.convert_to_tensor(x_test, dtype='float32')
    y_test = tf.convert_to_tensor(y_test, dtype='float32')

    my_init = keras.initializers.glorot_uniform(seed=1)
    model = keras.Sequential()
    model.add(Dense(units=5, activation="relu", name='InputLayer', input_shape=(770,), kernel_initializer=my_init))
    model.add(Dense(units=5, activation="relu", name='HiddenLayer'))
    model.add(Dense(units=1, activation='sigmoid', name='OuputLayer'))

    opt = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # # print(x_test.shape)
    # # print(y_test.shape)
    model.fit(x_train,
              y_train,
              validation_data=(x_dev, y_dev),
              epochs=50,  # Number of times the training vectors are used to update the weights.
              batch_size=16)  # For larger dataset, this helps in dividing the data into samples and train them.

    predicted = model.predict(x_test)
    y_predicted = np.round(predicted)
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy


def shuffle_split(X, y):
    X = np.array(X)
    y = np.array(y)

    sss1 = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
    k = 0
    accuracies = []

    for train_index, test_index in sss1.split(X, y):
        k += 1
        print('Split number: ' + str(k))
        X_train, X_more = X[train_index], X[test_index]
        y_train, y_more = y[train_index], y[test_index]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
        for train_index2, test_index2 in sss2.split(X_more, y_more):
            X_dev, X_test = X_more[train_index2], X_more[test_index2]
            y_dev, y_test = y_more[train_index2], y_more[test_index2]
            acc = MLP(X_train, y_train,
                      X_dev, y_dev,
                      X_test, y_test)
    accuracies.append(acc)
    return accuracies


def create_vectors(embeddings, music, emo):
    x = []
    y = []
    for ts in embeddings:
        x_vector = embeddings[ts] + music[ts]
        x.append(x_vector)
        y.append(emo[ts])

    ros = RandomOverSampler(random_state=0)
    x_resampled, y_resampled = ros.fit_resample(x, y)

    # print(y_resampled.count(1))
    # print(y_resampled.count(0))

    x_shuffle, y_shuffle = shuffle(x_resampled, y_resampled, random_state=0)
    # print(len(x_shuffle))
    # print(len(y_shuffle))
    return x_shuffle, y_shuffle


def get_data(embeddings_file, music_file, emo_file):
    ''' Read '''
    embeds = codecs.open(embeddings_file, 'r', 'utf-8')
    mus = codecs.open(music_file, 'r', 'utf-8')
    emos = codecs.open(emo_file, 'r', 'utf-8')
    mus_data = {}
    embeds_data = {}
    emos_data = {}
    for line in embeds:
        line = line.strip().split('\t')
        embeds_data[line[0]] = [float(i) for i in line[1:]]
    for line in mus:
        line = line.strip().split('\t')
        mus_data[line[0]] = [int(line[1]), int(line[2])]
    for line in emos:
        line = line.strip().split('\t')
        emos_data[line[0]] = int(line[1])
    return embeds_data, mus_data, emos_data


def main():
    music_file_path = 'data/music/music.tsv'
    embeddings_file_path = '../../BERTembeddings/layer13/embeddings_2sec.tsv'
    emo_file_path = '../../FORREST/to_bert/emo/sub-04_2sec.tsv'

    embeddings, music, emo = get_data(embeddings_file_path, music_file_path, emo_file_path)
    x, y = create_vectors(embeddings, music, emo)
    results = shuffle_split(x, y)
    print('RESULT')
    print(np.mean(results))


if __name__ == "__main__":
    main()