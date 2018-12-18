import re
import numpy as np
import hgtk
import pandas as pd
import keras
from keras.preprocessing import text, sequence
from keras.models import Model
from keras.layers import Input, Dense, GRU, LSTM
from keras.layers import Embedding
from sklearn.preprocessing import LabelEncoder
from layer_utils import Attention


class Continental:
    def __init__(self, path):
        self.path = path

    def read_excel(self):
        data = pd.read_excel(self.path, sheet_name='Sheet1')
        print(data.columns)
        data = data.filter(items=['고장원인전처리', '고장조치 라벨링', 'EquiGroup'])
        data = data.rename(index=str, columns={'고장원인전처리': 'X',
                                               '고장조치 라벨링': 'Y',
                                               'EquiGroup': 'equip'})
        print('equip counter:\n', data.equip.value_counts())
        return data

    def select_equip_by_name(self, data, equipname=None):
        self.equipname = equipname

        if self.equipname is None:
            self.equipname = 'all'
        else:
            data = data[data.equip.isin([self.equipname])]
        print('label counter:\n', data.Y.value_counts())
        return data

    def select_labels_by_index(self, data, topk):
        self.topk = topk
        label_cnt = data.Y.value_counts()
        col = label_cnt.index[:self.topk]
        data = data[data.Y.isin(col)]
        return data

    def _make_label(self, y, unique_label):
        # <make integer labels>
        int_idx = np.where(y == unique_label)[0].item()
        return int_idx

    def _char(self, x):
        # <character>
        x = hgtk.text.decompose(x)
        x = re.sub('\s+', ' ', x)
        return x

    def _sent2char(self, x):
        o = []
        for token in x:
            if hgtk.checker.is_hangul(token):
                token = hgtk.letter.decompose(token)
                token = ' '.join([token[0] + '/s0', token[1] + '/s1', token[2] + '/s2'])
            elif token == ' ':
                token = '<s>'
            o.append(token)
        o = ' '.join(o)
        o = re.sub('\s+', ' ', o)
        return o

    def insert_columns(self, data):
        # add some columns
        # labels = [self._make_label(x, unique_label=data.Y.unique()) for x in data.Y.values.tolist()]
        le = LabelEncoder()
        labels = le.fit_transform(data.Y)
        data.insert(data.shape[1], 'labels', labels)
        X_split = [self._char(x) for x in data.X.values.tolist()]
        data.insert(data.shape[1], 'X_split', X_split)

        return data

    def savedata(self, data):
        # <save dataset>
        data.to_pickle('datasets/subdata_E{}_C{}.pkl'.format(self.equipname, self.topk))

    def preprocessing(self, subdata):
        tokenizer = text.Tokenizer(filters='', char_level=True)
        tokenizer.fit_on_texts(subdata.X_split.values)
        # tokenizer.index_word[0] = '<pad>'
        print('number of chars:', len(tokenizer.index_word))  # except <pad>
        self.maxword = len(tokenizer.index_word)
        # tokenizer: char. level
        X = tokenizer.texts_to_sequences(subdata.X_split.values)
        self.maxlen = max([len(x) for x in X])
        print('max length: ', self.maxlen)
        # padding
        X = sequence.pad_sequences(X, maxlen=self.maxlen, truncating='pre', padding='pre')
        # one-hot label
        Y = keras.utils.to_categorical(subdata['labels'].values, num_classes=self.topk)
        print(Y.shape)
        return X, Y

    def model(self):
        x = Input(shape=(self.maxlen,), dtype='int32')
        h = Embedding(self.maxword + 1, 64,
                      input_length=self.maxlen,
                      mask_zero=True)(x)
        h = LSTM(128, return_sequences=False, unroll=True)(h)
        # h = Attention(self.maxlen)(h)
        # h = Dense(64, activation='relu')(h)
        y = Dense(self.topk, activation='softmax')(h)
        model = Model(x, y)

        opt = keras.optimizers.Adam()
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=[keras.metrics.categorical_accuracy])
        model.summary()
        return model


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # make dataset
    path = 'datasets/PM_fail&solution_data.xlsx'
    cont = Continental(path)
    topk = 2

    data = cont.read_excel()
    data = cont.select_equip_by_name(data, equipname=None)
    data = cont.select_labels_by_index(data, topk)
    data = cont.insert_columns(data)
    cont.savedata(data)

    # data = pd.read_pickle('datasets/subdata_Eall_C2.pkl')
    cont.topk=topk
    # modeling
    X, Y = cont.preprocessing(data)
    model = cont.model()

    # train
    chkpoint = keras.callbacks.ModelCheckpoint('history/aa.hdf5', save_best_only=True, save_weights_only=True)
    hist = model.fit(X, Y, epochs=150, batch_size=256,
                     shuffle=True, validation_split=0.2,
                     verbose=2, callbacks=[chkpoint])

    # '베큠로더 픽업에러및 spi 틀어짐에러발생 픽업포지션재설정 spi 재테스트'