#%%
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = np.load("../dataSC-embeddings/embeddings_0_test.npy")  # (N, 320)
y = np.load("../dataSC-embeddings/labels_0_test.npy").squeeze()      # (N,)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
clf = LogisticRegression(max_iter=5000)
clf.fit(X_train, y_train)
print("Train acc:", clf.score(X_train, y_train))
print("Val acc:", clf.score(X_val, y_val))


#%%
import pandas as pd
from pathlib import Path
import numpy as np

#check if any other clasification label makes sense...
csv = pd.read_csv("../../SCAPISdata/SCAPIS-DATA-PETITION-659-20241024.csv").set_index("Subject")
trainvalas = [i.name.split(".")[0][2:] for i in Path("../dataSC/train").glob("*.nii.gz")]
testvalas = [i.name.split(".")[0][2:] for i in Path("../dataSC/test").glob("*.nii.gz")]
valvalas = [i.name.split(".")[0][2:] for i in Path("../dataSC/val").glob("*.nii.gz")]

traincsv = csv[csv.index.isin(trainvalas)]
testcsv = csv[csv.index.isin(testvalas)]
valcsv = csv[csv.index.isin(valvalas)]
#%%

suma=testcsv[["CACS_RCA","CACS_Circumfl","CACS_LAD_LM"]].sum(axis=1)
suma = pd.Series(suma).dropna()
suma[(suma>10) & (suma<1000)].hist(bins=20)

(len(suma), len(suma[suma==0]), len(suma[suma>100]))
# %%
