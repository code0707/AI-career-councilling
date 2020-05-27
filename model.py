import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df = pd.read_csv("dataset.csv")
mapping = pd.read_csv("mapping.csv")
X = df.drop(["Suggested Job Role"],axis=1)
y = df["Suggested Job Role"]
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

y_pred = clf.predict([[60,40,100,45,20,10,78,49,90,100,18,14,94,100]])
for i in mapping["Suggested Job Role"]:
    if(y_pred == i):
        print(mapping["Mapping Suggested Job Role"][i])