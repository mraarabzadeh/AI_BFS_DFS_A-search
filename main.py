import numpy as np
import pandas as pd
from sklearn import tree
MAX_RANGE = 240
NONE_TARGET_FEACHER = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal',]
data = pd.read_csv('data.csv')
x = data[NONE_TARGET_FEACHER][:MAX_RANGE]
y = data['target'][:MAX_RANGE]
def decision_tree():
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x,y)
    result = clf.predict(data[NONE_TARGET_FEACHER][MAX_RANGE:])
    acc = 0
    for i in range(MAX_RANGE, 303):
        if result[i - MAX_RANGE] == data['target'][i]:
            acc += 1
    return acc
def make_random_list():
    res = []
    for i in range(5):
        res.append(np.random.randint(low=0, high=MAX_RANGE, size=150))
    return res




    
# print(decision_tree())
