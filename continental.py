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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils import class_weight


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
        """Deprecated method"""
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
        # add label
        le = LabelEncoder()
        labels = le.fit_transform(data.Y)
        data.insert(data.shape[1], 'labels', labels)

        # add one-hot equip
        le = LabelEncoder()
        equip = le.fit_transform(data.equip)
        data.insert(data.shape[1], 'equip_index', equip)
        self.num_euip = len(data.equip_index.unique())

        # add char-level inputs
        X_split = [self._char(x) for x in data.X.values.tolist()]
        data.insert(data.shape[1], 'X_split', X_split)
        return data

    def savedata(self, data):
        # <save dataset>
        data.to_pickle('datasets/subdata_E{}_C{}.pkl'.format(self.equipname, self.topk))

    def preprocessing(self, subdata):
        tokenizer = text.Tokenizer(filters='', char_level=True)
        tokenizer.fit_on_texts(subdata.X_split.values)
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
        X_equip = keras.utils.to_categorical(subdata['equip_index'].values)
        return X, X_equip, Y

    def model(self):
        x = Input(shape=(self.maxlen,), dtype='int32')
        h = Embedding(self.maxword + 1, 64,
                      input_length=self.maxlen,
                      mask_zero=True)(x)
        h = LSTM(128, return_sequences=True, unroll=True)(h)
        h = Attention(self.maxlen)(h)
        y = Dense(self.topk, activation='softmax')(h)
        model = Model(x, y)

        opt = keras.optimizers.Adam()
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=[keras.metrics.categorical_accuracy])
        model.summary()
        return model

    def model_equip(self):
        x = Input(shape=(self.maxlen,), dtype='int32')
        x_equip = Input(shape=(self.num_euip,), dtype='float32')
        h_equip = Dense(64, activation='relu')(x_equip)
        h = Embedding(self.maxword + 1, 64,
                      input_length=self.maxlen,
                      mask_zero=True)(x)
        h = LSTM(128, return_sequences=True, unroll=True)(h)
        h = Attention(self.maxlen)(h)
        h = keras.layers.concatenate([h, h_equip])
        y = Dense(self.topk, activation='softmax')(h)
        model = Model([x, x_equip], y)

        opt = keras.optimizers.Adam()

        def top3_acc(y_true, y_pred):
            return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

        def top5_acc(y_true, y_pred):
            return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['acc', top3_acc, top5_acc])
        model.summary()
        return model


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # make dataset
    path = 'datasets/PM_fail&solution_data.xlsx'
    cont = Continental(path)
    topk = 100
    data = cont.read_excel()
    data = cont.select_equip_by_name(data, equipname=None)
    data = cont.select_labels_by_index(data, topk)
    data = cont.insert_columns(data)
    cont.savedata(data)
    # data = pd.read_pickle('datasets/subdata_Eall_C2.pkl')
    # cont.topk=topk

    # modeling
    X, X_equip, Y = cont.preprocessing(data)
    # model = cont.model()
    model = cont.model_equip()

    # train
    train_indices, test_indices = train_test_split(
        np.arange(len(X)), test_size=0.2, shuffle=True,
        random_state=123123
    )
    X_train, X_equip_train, Y_train = X[train_indices], X_equip[train_indices], Y[train_indices]
    X_test, X_equip_test, Y_test = X[test_indices], X_equip[test_indices], Y[test_indices]

    model_path = 'history/model_top100.hdf5'

    # set class weight
    cls_wgt = class_weight.compute_class_weight(class_weight='balanced',
                                                classes=np.unique(np.argmax(Y_train, axis=-1)),
                                                y=np.argmax(Y_train, axis=-1))
    chkpoint = keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True)
    hist = model.fit([X_train, X_equip_train], Y_train, epochs=150, batch_size=256,
                     shuffle=True, validation_split=0.1, class_weight=cls_wgt,
                     verbose=2, callbacks=[chkpoint])
    np.save('history/hist_top100.npy', hist.history)

    # evaluation
    model.load_weights(model_path)
    Y_pred = model.predict([X_test, X_equip_test], batch_size=256)
    cm = confusion_matrix(np.argmax(Y_test, axis=-1), np.argmax(Y_pred, axis=-1))
    print(classification_report(np.argmax(Y_test, axis=-1), np.argmax(Y_pred, axis=-1)))

    score = model.evaluate([X_test, X_equip_test], Y_test, batch_size=256, verbose=2)
    # top10
    # [1.0131853095631607, 0.6542401817893874, 0.9005406946707125]
    '''
              0       0.57      0.64      0.60      2089
              1       0.81      0.61      0.70       691
              2       0.89      0.88      0.89       836
              3       0.50      0.34      0.40       764
              4       0.68      0.75      0.71      2554
              5       0.72      0.68      0.70      2201
              6       0.59      0.51      0.55      2554
              7       0.56      0.82      0.67      1147
              8       0.76      0.85      0.80       754
              9       0.57      0.30      0.40       466
    avg / total       0.66      0.65      0.65     14056
    '''
    # top 100
    # loss, top1 acc, top3 acc, top5 acc
    # [1.9646599202466826, 0.4786316862890516, 0.7061740984076353, 0.7994808566197856]
    '''
                 precision    recall  f1-score   support
          0       0.73      0.72      0.72        71
          1       0.46      0.53      0.50      2161
          2       0.29      0.18      0.22        68
          3       0.40      0.10      0.16        61
          4       0.00      0.00      0.00        24
          5       0.20      0.07      0.10       247
          6       0.00      0.00      0.00        14
          7       0.19      0.14      0.16        21
          8       0.40      0.12      0.18        51
          9       0.00      0.00      0.00        26
         10       0.66      0.58      0.62       122
         11       0.57      0.08      0.15        48
         12       0.47      0.32      0.38        56
         13       0.73      0.22      0.33        37
         14       0.42      0.57      0.48        40
         15       0.65      0.57      0.61       633
         16       0.27      0.02      0.04       132
         17       0.16      0.10      0.12        29
         18       0.77      0.83      0.80       869
         19       0.29      0.42      0.34       353
         20       0.33      0.07      0.12        56
         21       0.00      0.00      0.00        30
         22       0.00      0.00      0.00        25
         23       0.00      0.00      0.00        60
         24       0.00      0.00      0.00        29
         25       0.17      0.03      0.05        34
         26       0.00      0.00      0.00        38
         27       0.00      0.00      0.00        21
         28       0.08      0.01      0.02       101
         29       0.50      0.03      0.06        61
         30       0.17      0.02      0.03        55
         31       0.67      0.11      0.18        38
         32       0.62      0.14      0.22        37
         33       0.64      0.27      0.38        67
         34       0.41      0.06      0.10       425
         35       0.00      0.00      0.00        36
         36       0.33      0.21      0.26        19
         37       0.34      0.28      0.30       156
         38       0.83      0.64      0.72        94
         39       0.38      0.15      0.22        33
         40       0.29      0.13      0.18       374
         41       0.33      0.37      0.35       796
         42       0.31      0.26      0.28       350
         43       0.42      0.45      0.44        49
         44       0.51      0.73      0.60      2621
         45       0.51      0.66      0.58       441
         46       0.35      0.21      0.26        53
         47       0.27      0.37      0.32        75
         48       0.40      0.08      0.13        26
         49       0.33      0.04      0.07        27
         50       0.29      0.06      0.09       181
         51       0.10      0.05      0.07       105
         52       0.35      0.25      0.29        24
         53       0.90      0.11      0.20        81
         54       0.22      0.11      0.14        19
         55       0.33      0.04      0.07        25
         56       0.38      0.58      0.46        86
         57       0.22      0.14      0.17        43
         58       0.45      0.35      0.39        84
         59       0.47      0.47      0.47        19
         60       0.32      0.16      0.21        63
         61       0.58      0.49      0.53        89
         62       0.00      0.00      0.00        31
         63       0.00      0.00      0.00        31
         64       0.50      0.70      0.59       251
         65       0.11      0.03      0.05        33
         66       0.00      0.00      0.00        26
         67       0.00      0.00      0.00        27
         68       0.54      0.63      0.58      2186
         69       0.26      0.13      0.17       123
         70       0.45      0.44      0.44      2558
         71       0.00      0.00      0.00        89
         72       0.33      0.03      0.06        33
         73       0.51      0.49      0.50       162
         74       0.27      0.13      0.18       252
         75       0.00      0.00      0.00        17
         76       0.00      0.00      0.00        24
         77       0.00      0.00      0.00        29
         78       0.00      0.00      0.00        28
         79       0.27      0.06      0.10        65
         80       0.00      0.00      0.00        26
         81       0.12      0.06      0.08        93
         82       0.00      0.00      0.00        30
         83       0.50      0.07      0.12        90
         84       0.21      0.15      0.18        52
         85       0.00      0.00      0.00        40
         86       0.00      0.00      0.00        32
         87       0.30      0.20      0.24       150
         88       0.41      0.73      0.53      1134
         89       0.33      0.05      0.09        79
         90       0.52      0.28      0.36        50
         91       0.70      0.83      0.76       748
         92       0.50      0.77      0.61        93
         93       0.30      0.24      0.26       268
         94       0.23      0.25      0.24        51
         95       0.36      0.42      0.39       443
         96       0.18      0.13      0.15        15
         97       0.00      0.00      0.00        29
         98       0.25      0.03      0.05        40
         99       0.00      0.00      0.00        37
avg / total       0.45      0.48      0.44     21574
    '''