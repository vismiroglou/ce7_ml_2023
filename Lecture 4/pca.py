from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

trn5 = pd.read_csv('Data/train5.txt', sep='\s+', header= None)
trn6 = pd.read_csv('Data/train6.txt', sep='\s+', header= None)
trn8 = pd.read_csv('Data/train8.txt', sep='\s+', header= None)

tst5 = pd.read_csv('Data/test5.txt', sep='\s+', header= None)
tst6 = pd.read_csv('Data/test6.txt', sep='\s+', header= None)
tst8 = pd.read_csv('Data/test8.txt', sep='\s+', header= None)

trn = pd.concat((trn5, trn6, trn8), axis =0)
trn_x = trn.iloc[:, :-1]
trn_y = trn.iloc[:, -1:]
del trn

tst = pd.concat((tst5, tst6, tst8), axis =0)
tst_x = tst.iloc[:, :-1]
tst_y = tst.iloc[:, -1:]
del tst

pca = PCA(n_components = 2)
pca.fit(trn)
trn2d = pca.transform(trn)

print(trn2d.shape)
