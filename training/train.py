# Larger LSTM Network to Generate Text for Alice in Wonderland
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/include/

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Activation
from keras.callbacks import ModelCheckpoint, LambdaCallback, TensorBoard
from keras.utils import np_utils
import random
import numpy as np
import sys
import os
import unidecode
import string

# load ascii text and covert to lowercase
filename = "data/eminem.txt"

charset = set(string.digits + string.letters + string.punctuation + "\n ")
raw_text = filter(lambda x: x in charset, unidecode.unidecode(open(filename, 'r').read().decode('utf-8')))
chars = sorted(list(charset))

n_chars = len(raw_text)
print 'total chars:', len(chars) 
print chars

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# prepare the dataset of input to output pairs encoded as integers
seq_length = 48


model_name = 'eminem-128-128'

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(seq_length, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint

filepath = "models/" + model_name + "/{epoch:02d}-{loss:.4f}.hdf5"

if not os.path.exists(os.path.dirname(filepath)):
    os.makedirs(os.path.dirname(filepath))

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def sample_text(epoch, logs):
    start_index = random.randint(0, len(raw_text) - seq_length - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print '----- diversity:', diversity

        generated = ''
        sentence = raw_text[start_index: start_index + seq_length]
        generated += sentence
        sys.stdout.write(generated + '//')

        for i in range(200):

            x = np.zeros((1, seq_length, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print ""

def generate_examples():
    while True:
        dataX = []
        dataY = []

        # cut the text in semi-redundant sequences of seq_length characters
        sentences = []
        next_chars = []

        sequential_chunk = 10

        for j in range(128):
            i_base = random.randrange(0, n_chars - seq_length - sequential_chunk - 1, 1)

            for i in range(i_base, i_base + sequential_chunk, 3):
                sentences.append(raw_text[i: i + seq_length])
                next_chars.append(raw_text[i + seq_length])

        X = np.zeros((len(sentences), seq_length, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1

        yield (X, y)


callbacks_list = [ checkpoint, LambdaCallback(on_epoch_begin = sample_text), TensorBoard('logs/' + model_name) ]


# model.load_weights("weights-full-alpha-00-1.9680.hdf5")
# fit the model
model.fit_generator(generate_examples(), steps_per_epoch=5000, epochs=500, callbacks=callbacks_list)
