from continental import Continental
from keras.callbacks import ModelCheckpoint
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# make dataset
path = 'datasets/PM_fail&solution_data.xlsx'
cont = Continental(path)
topk = 2

data = cont.read_excel()
# data = cont.select_equip_by_name(data, [e])
data = cont.select_labels_by_index(data, topk)
data = cont.get_subdata(data)
cont.savedata(data)

# modeling
X, Y = cont.preprocessing(data)
model = cont.model()

# train
chkpoint = ModelCheckpoint('history/aa.hdf5', save_best_only=True, save_weights_only=True)
hist = model.fit(X, Y, epochs=150, batch_size=128,
                shuffle=True, validation_split=0.2,
                verbose=1)

for e in ['Mounter', 'ICT', 'Robot Soldering']:
    data = cont.read_excel()
    # data = cont.select_equip_by_name(data, [e])
    data = cont.select_labels_by_index(data, topk)
    data = cont.get_subdata(data)
    cont.savedata(data)

    # modeling
    X, Y = cont.preprocessing(data)
    model = cont.model()

    # train
    chkpoint = ModelCheckpoint('history/aa.hdf5', save_best_only=True, save_weights_only=True)
    hist = model.fit(X, Y, epochs=150, batch_size=128,
                     shuffle=True, validation_split=0.2,
                     verbose=1)
