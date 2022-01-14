from modnet.preprocessing import MODData
from modnet.models import MODNetModel
import numpy as np
import os
import copy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import KFold
from modnet.preprocessing import MODData

import tensorflow as tf
import random

seed = 1234
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def shuffle_MD(data,random_state=10):
    data = copy.deepcopy(data)
    ids = data.df_targets.sample(frac=1,random_state=random_state).index
    data.df_featurized = data.df_featurized.loc[ids]
    data.df_targets = data.df_targets.loc[ids]
    data.df_structure = data.df_structure.loc[ids]

    return data

def MDKsplit(data,n_splits=5,random_state=10):
    data = shuffle_MD(data,random_state=random_state)
    ids = np.array(data.structure_ids)
    kf = KFold(n_splits=n_splits,shuffle=True,random_state=random_state)
    folds = []
    for train_idx, val_idx in kf.split(ids):
        data_train = MODData(data.df_structure.iloc[train_idx]['structure'].values,data.df_targets.iloc[train_idx].values,target_names=data.df_targets.columns,structure_ids=ids[train_idx])
        data_train.df_featurized = data.df_featurized.iloc[train_idx]
        #data_train.optimal_features = data.optimal_features

        data_val = MODData(data.df_structure.iloc[val_idx]['structure'].values,data.df_targets.iloc[val_idx].values,target_names=data.df_targets.columns,structure_ids=ids[val_idx])
        data_val.df_featurized = data.df_featurized.iloc[val_idx]
        #data_val.optimal_features = data.optimal_features

        folds.append((data_train,data_val))

    return folds

def MD_append(md,lmd):
    md = copy.deepcopy(md)
    for m in lmd:
        md.df_structure = md.df_structure.append(m.df_structure)
        md.df_targets = md.df_targets.append(m.df_targets)
        md.df_featurized = md.df_featurized.append(m.df_featurized)
    return md

md_exp = MODData.load('exp_gap_all')
md_exp.df_targets.columns = ['gap']
md_pbe = MODData.load('pbe_gap.zip')
md_pbe.df_targets.columns = ['gap']
md_hse = MODData.load('hse_gap.zip')
md_hse.df_targets.columns = ['gap']
md_gllb=MODData.load("gllb_gap.zip")
md_gllb.df_targets.columns = ['gap']
md_scan=MODData.load("scan_md_new_from_pp.zip")
md_scan.df_targets.columns = ['gap']

k = 2
random_state = 202010
folds = MDKsplit(md_exp,n_splits=k,random_state=random_state)
maes = np.ones(5)
for i,f in enumerate(folds):
    train = f[0]
    test = f[1]
    fpath = 'train_folds/train_{}_{}'.format(random_state,i+1)
    if os.path.exists(fpath):
        train = MODData.load(fpath)
        train.df_targets.columns=['gap']
    else:
        train.feature_selection(n=-1, n_jobs=10)
        train.save(fpath)

    # assure no overlap
    assert len(set(train.df_targets.index).intersection(set(test.df_targets.index))) == 0

    model = MODNetModel([[['gap']]],{'gap':1})
    # rlr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=20, verbose=1, mode="auto", min_delta=0)
    es = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10, verbose=1, mode="auto", baseline=None,restore_best_weights=True)
    model.fit(train, val_fraction=0.2, lr=0.01, epochs = 1000, batch_size = 64, loss='mae', callbacks=[es], verbose=1)

    pred = model.predict(test)
    true = test.df_targets
    error = pred-true
    error = error.drop(pred.index[((pred['gap']).abs()>20)]) # drop unrealistic values: happens extremely rarely
    mae = np.abs(error.values).mean()
    maes[i] = mae
    model.save('out/MODNet_E_only_{}'.format(i+1))
print(maes)
