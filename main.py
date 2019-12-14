import numpy as np
import pandas as pd
from sklearn import tree
MAX_RANGE = 240
NONE_TARGET_FEACHER = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal',]
data = pd.read_csv('data.csv')
data = data.sample(frac=1).reset_index(drop=True)

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

def bagging(feachers = NONE_TARGET_FEACHER):
    l = []
    rand = make_random_list()
    for i in range(5):
        l.append(tree.DecisionTreeClassifier(max_depth = 13, max_features = 12))
        l[i] = l[i].fit(data[feachers].iloc[rand[i]], data['target'].iloc[rand[i]])
    return l
def random_forrest():
    l = []
    rand = make_random_list()
    random_feacher = [[],[],[],[],[]]
    for i in range(5):
        for j in range(5):
            random_feacher_ind = np.random.randint(low = 0, high=len(NONE_TARGET_FEACHER), size = 5)
            random_feacher[i].append(NONE_TARGET_FEACHER[random_feacher_ind[i]])
        l.append(tree.DecisionTreeClassifier(max_features = 5))
        l[i] = l[i].fit(data[random_feacher[i]].iloc[rand[i]], data['target'].iloc[rand[i]])
        random_feacher_ind = np.random.randint(low = 0, high=len(NONE_TARGET_FEACHER), size = 5)
    return l, random_feacher
def eval_ensemmble(bagging_trees, feachers = NONE_TARGET_FEACHER):
    answers = []
    for i in range(5):
        answers.append(bagging_trees[i].predict(data[feachers][MAX_RANGE:]))
    for i in range(4):
        for j in range(len(answers[0])):
            answers[0][j] += answers[i+1][j]
    answers[0] = list(map(lambda x: 1 if x > 2 else 0, answers[0]))
    return calc_acc(answers[0])
def eval_random_forest(bagging_trees, feachers):
    answers = []
    for i in range(5):
        answers.append(bagging_trees[i].predict(data[feachers[i]][MAX_RANGE:]))
    for i in range(4):
        for j in range(len(answers[0])):
            answers[0][j] += answers[i+1][j]
    answers[0] = list(map(lambda x: 1 if x > 2 else 0, answers[0]))
    return calc_acc(answers[0])
def find_important_feacher():
    for i in range(len(NONE_TARGET_FEACHER)):
        test_feacher = NONE_TARGET_FEACHER[0:i] + NONE_TARGET_FEACHER[i+1:MAX_RANGE]
        trees = bagging(test_feacher)
        res =  eval_ensemmble(trees, feachers=test_feacher)
        if res <45 :
            print(NONE_TARGET_FEACHER[i], '\t -->', res)
bagging_trees = bagging()
random_forrests, feachers_selected = random_forrest()
print(feachers_selected)
print(eval_ensemmble(bagging_trees))
print(eval_random_forest(random_forrests, feachers_selected))
print(decision_tree())
# for i in range(100):
#     print('--------new test------------')
#     find_important_feacher()
#     print('\n\n')
# print(decision_tree())
