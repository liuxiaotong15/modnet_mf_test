from modnet.preprocessing import MODData
from modnet.models import MODNetModel
import numpy as np
import os
import copy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import KFold
from modnet.preprocessing import MODData

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

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
        md.df_structure.append(m.df_structure)
        md.df_targets.append(m.df_targets)
        md.df_featurized.append(m.df_featurized)
    
    md = shuffle_MD(md)
    return md

md_exp = MODData.load('exp_gap_all')
md_exp.df_targets.columns = ['gap']
md_pbe = MODData.load('pbe_gap.zip')
md_pbe.df_targets.columns = ['gap']
md_hse = MODData.load('hse_gap.zip')
md_hse.df_targets.columns = ['gap']
md_gllb=MODData.load("gllb_gap.zip")
md_gllb.df_targets.columns = ['gap']
md_scan=MODData.load("scan_md.zip")
md_scan.df_targets.columns = ['gap']

k = 2
random_state = 202010
folds = MDKsplit(md_exp,n_splits=k,random_state=random_state)
maes_ph1 = np.ones(5)
maes_ph2 = np.ones(5)
maes_ph3 = np.ones(5)
maes_ph4 = np.ones(5)
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

    #phase 1
    md = MD_append(train,[md_pbe,md_hse,md_gllb,md_scan])

    model = MODNetModel([[['gap']]],{'gap':1})
    es = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10, verbose=1, mode="auto", baseline=None,restore_best_weights=True)
    model.fit(md, val_fraction=0.2, lr=0.01, epochs = 1000, batch_size = 64, loss='mae', callbacks=[es], verbose=1)

    model.save('out/MODNet_onion_{}_ph1'.format(i+1))

    pred = model.predict(test)
    true = test.df_targets
    error = pred-true
    error = error.drop(pred.index[((pred['gap']).abs()>20)]) # drop unrealistic values: happens extremely rarely
    mae = np.abs(error.values).mean()
    print('mae_ph1')
    print(mae)
    maes_ph1[i] = mae

    #phase 2
    md = MD_append(train,[md_pbe,md_hse,md_scan])

    es = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10, verbose=1, mode="auto", baseline=None,restore_best_weights=True)
    model.fit(md, val_fraction=0.2, lr=0.01, epochs = 1000, batch_size = 64, loss='mae', callbacks=[es], verbose=1)

    model.save('out/MODNet_onion_{}_ph2'.format(i+1))

    pred = model.predict(test)
    true = test.df_targets
    error = pred-true
    error = error.drop(pred.index[((pred['gap']).abs()>20)]) # drop unrealistic values: happens extremely rarely
    mae = np.abs(error.values).mean()
    print('mae_ph2')
    print(mae)
    maes_ph2[i] = mae

    #phase 3
    md = MD_append(train,[md_hse, md_scan])

    es = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10, verbose=1, mode="auto", baseline=None,restore_best_weights=True)
    model.fit(md, val_fraction=0.2, lr=0.01, epochs = 1000, batch_size = 64, loss='mae', callbacks=[es], verbose=1)

    model.save('out/MODNet_onion_{}_ph3'.format(i+1))

    pred = model.predict(test)
    true = test.df_targets
    error = pred-true
    error = error.drop(pred.index[((pred['gap']).abs()>20)]) # drop unrealistic values: happens extremely rarely
    mae = np.abs(error.values).mean()
    print('mae_ph3')
    print(mae)
    maes_ph3[i] = mae

    # phase 4
    md = MD_append(train,[md_hse])
    # rlr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=20, verbose=1, mode="auto", min_delta=0)
    es = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10, verbose=1, mode="auto", baseline=None,restore_best_weights=True)
    model.fit(md, val_fraction=0.2, lr=0.01, epochs = 1000, batch_size = 64, loss='mae', callbacks=[es], verbose=1)

    model.save('out/MODNet_onion_{}_ph4'.format(i+1))

    pred = model.predict(test)
    true = test.df_targets
    error = pred-true
    error = error.drop(pred.index[((pred['gap']).abs()>20)]) # drop unrealistic values: happens extremely rarely
    mae = np.abs(error.values).mean()
    print('mae_ph4')
    print(mae)
    maes_ph4[i] = mae

    # phase 5
    # rlr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=20, verbose=1, mode="auto", min_delta=0)
    es = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10, verbose=1, mode="auto", baseline=None,restore_best_weights=True)
    model.fit(train, val_fraction=0.2, lr=0.01, epochs = 1000, batch_size = 64, loss='mae', callbacks=[es], verbose=1)

    model.save('out/MODNet_onion_{}_ph5'.format(i+1))

    pred = model.predict(test)
    true = test.df_targets
    error = pred-true
    error = error.drop(pred.index[((pred['gap']).abs()>20)]) # drop unrealistic values: happens extremely rarely
    mae = np.abs(error.values).mean()
    print('mae')
    print(mae)
    maes[i] = mae

print(maes_ph1)
print(maes_ph2)
print(maes_ph3)
print(maes_ph4)
print(maes)
