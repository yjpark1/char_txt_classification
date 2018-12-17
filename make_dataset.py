import re
import numpy as np
import hgtk
import pandas as pd
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data = pd.read_excel('datasets/PM_fail&solution_data.xlsx', sheet_name='Sheet1')
print(data.columns)
data = data.filter(items=['고장원인전처리', '고장조치라벨링', 'EquiGroup'])
data = data.rename(index=str, columns={'고장원인전처리': 'X',
                                       '고장조치라벨링': 'Y',
                                       'EquiGroup': 'equip'})
label_cnt = data.Y.value_counts()
print(label_cnt)

topx = 2
label_cnt.index[:topx]
""" Top5 labels
1) 수동 버튼 조작/점검/리셋 및 임시 조치 후 점검 요청    12965
2) 장비/시스템 점검 및 리셋                    12500
3) 장비 청소 및 부품 세척                     11100
4) PC 점검 및 수리                        10610
5) 프로브 점검 및 교체                        5747
"""
subdata = data[data.Y.isin(label_cnt.index[:topx])]

# <make integer labels>
def make_label(y, unique_label=label_cnt.index[:topx].values):
    for idx, lbl in enumerate(unique_label):
        if y in lbl:
            break
    return idx

labels = [make_label(x) for x in subdata.Y.values.tolist()]
subdata['labels'] = labels

equip = [make_label(x, unique_label=subdata.equip.unique()) for x in subdata.equip.values.tolist()]
subdata['X_add'] = equip
# subdata.loc[:,'X_add'] = equip

# <character>
def char(x):
    x = hgtk.text.decompose(x)
    return re.sub('\s+', '↔', x)

X_split = [char(x) for x in subdata.X.values.tolist()]
subdata['X_split'] = X_split

# <save dataset>
subdata.to_pickle('datasets/subdata_top{}.pkl'.format(topx))