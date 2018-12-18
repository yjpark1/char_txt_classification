import re
import numpy as np
import hgtk
import pandas as pd


class MakeData:
    def __init__(self, path):
        self.path = path

    def read_excel(self):
        data = pd.read_excel(self.path, sheet_name='Sheet1')
        print(data.columns)
        data = data.filter(items=['고장원인전처리', '고장조치라벨링', 'EquiGroup'])
        data = data.rename(index=str, columns={'고장원인전처리': 'X',
                                            '고장조치라벨링': 'Y',
                                            'EquiGroup': 'equip'})        
        print('equip counter:', data.equip.value_counts())        
        return data
        
    def select_equip_by_name(self, data, equipname):
        self.equipname = equipname
        data = data[data.equip.isin(self.equipname)]
        print('label counter:', data.Y.value_counts())
        return data

    def select_labels_by_index(self, data, col_idx):
        self.col_idx = col_idx
        label_cnt = data.Y.value_counts()
        col = label_cnt.index[self.col_idx]
        return data[data.Y.isin(col)]
    
    def _make_label(self, y, unique_label):
        # <make integer labels>        
        int_idx = np.where(y == unique_label)[0].item()
        return int_idx
    
    def _char(self, x):
        # <character>
        x = hgtk.text.decompose(x)
        return re.sub('\s+', '↔', x)
    
    def get_subdata(self, data):
        # add some columns
        labels = [self.make_label(x, unique_label=data.Y.unique()) for x in data.Y.values.tolist()]
        equip = [self.make_label(x, unique_label=data.equip.unique()) for x in data.equip.values.tolist()]
        data.insert(3, 'labels', labels)
        data.insert(4, 'X_add', equip)        
        # character-level decomposition
        X_split = [self.char(x) for x in data.X.values.tolist()]        
        data.insert(5, 'X_split', X_split)
        return data
    
    def save(self, data)
        # <save dataset>
        data.to_pickle('datasets/subdata_E{}_C{}.pkl'.format(self.equipname, self.col_idx))
        
    
if __name__ == '__main__':
    path = 'datasets/PM_fail&solution_data.xlsx'
    makedata = MakeData(path)
    data = makedata.read_excel()
    data = makedata.select_equip_by_name(['Mounter'])
    data = makedata.select_labels_by_index(0:4)
    data = makedata.get_subdata(data)
    makedata.save(data)
