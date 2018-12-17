import numpy as np
import pandas as pd
import os
from keras.preprocessing import text, sequence
import keras
from keras.models import Model
from keras.layers import Input, Dense, GRU
from keras.layers import Embedding
from layer_utils import Attention


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

topx = 5
subdata = pd.read_pickle('datasets/subdata_top{}.pkl'.format(topx))

tokenizer = text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(subdata.X_split.values)
tokenizer.index_word[0] = '<pad>'
print('number of chars:', len(tokenizer.index_word))  # except <pad>

# tokenizer: char. level
X = tokenizer.texts_to_sequences(subdata.X_split.values)
# padding
X = sequence.pad_sequences(X, maxlen=120, truncating='post', padding='pre')
# one-hot label
Y = keras.utils.to_categorical(subdata['labels'].values)
print(Y.shape)

# define model
x = Input(shape=(120,), dtype='int32')
h = Embedding(len(tokenizer.index_word) + 1, 16,
              input_length=120,
              trainable=True)(x)
h = GRU(64, return_sequences=True, unroll=True,
        dropout=0.25, recurrent_dropout=0.25)(h)
h = Attention(120)(h)
y = Dense(topx, activation='softmax')(h)
model = Model(x, y)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# train
hist = model.fit(X, Y, epochs=150, batch_size=128,
                 shuffle=True, validation_split=0.2,
                 verbose=2)

