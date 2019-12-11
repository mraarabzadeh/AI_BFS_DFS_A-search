import numpy as np
import pandas as pd
from sklearn import tree
MAX_RANGE = 240
NONE_TARGET_FEACHER = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal',]
data = pd.read_csv('data.csv')
data = data.sample(frac=1).reset_index(drop=True)
print(data)
def calc_acc(result):
    acc = 0
    for i in range(MAX_RANGE, 303):
        if result[i - MAX_RANGE] == data['target'][i]:
            acc += 1
    return acc
def decision_tree():
    x = data[NONE_TARGET_FEACHER][:MAX_RANGE]
    y = data['target'][:MAX_RANGE]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x,y)
    result = clf.predict(data[NONE_TARGET_FEACHER][MAX_RANGE:])    
    return calc_acc(result)
def make_random_list():
    res = []
    for i in range(5):
        res.append(np.random.randint(low=0, high=MAX_RANGE, size=150))
    return res

def bagging():
    l = []
    rand = make_random_list()
    for i in range(5):
        l.append(tree.DecisionTreeClassifier())
        l[i] = l[i].fit(data[NONE_TARGET_FEACHER].iloc[rand[i]], data['target'].iloc[rand[i]])
    return l
def random_forrest():
    l = []
    rand = make_random_list()
    random_feacher_ind = np.random.randint(low = 0, high=len(NONE_TARGET_FEACHER), size = 5)
    random_feacher = []
    for i in range(5):
        random_feacher.append(NONE_TARGET_FEACHER[random_feacher_ind[i]])
    for i in range(5):
        l.append(tree.DecisionTreeClassifier())
        l[i] = l[i].fit(data[random_feacher].iloc[rand[i]], data['target'].iloc[rand[i]])
    return l, random_feacher    
def eval_bagging(bagging_trees, feachers = NONE_TARGET_FEACHER):
    answers = []
    for i in range(5):
        answers.append(bagging_trees[i].predict(data[feachers][MAX_RANGE:]))
    for i in range(4):
        for j in range(len(answers[0])):
            answers[0] += answers[i+1][j]
    answers[0] = list(map(lambda x: 1 if x > 2 else 0, answers[0]))
    return calc_acc(answers[0])
# def eval_random_forrest(trees, feachers):
#     answers = []
#     for i in range(5):
#         answers.append(bagging_trees[i].predict(data[feachers][MAX_RANGE:]))
#     for i in range(4):
#         for j in range(len(answers[0])):
#             answers[0] += answers[i+1][j]
#     answers[0] = list(map(lambda x: 1 if x > 2 else 0, answers[0]))
#     return calc_acc(answers[0])
bagging_trees = bagging()
random_forrest, feachers_selected = random_forrest()
print(eval_bagging(bagging_trees))
print(eval_bagging(random_forrest, feachers=feachers_selected))
print(decision_tree())
# print(decision_tree())
