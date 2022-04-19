from dataclasses import replace
from math import log2
import math
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from collections import Counter
import matplotlib.pyplot as plt
from statistics import mean


#RUN THIS FILE USING ANY PYTHON IDE, PLEASE GO TO LINE 178 FOR LOADING DATASETS, make sure the datasets are in a 'datasets' folder

class TreeNodeInterface():
    """Simple "interface" to ensure both types of tree nodes must have a classify() method."""
    def classify(self, example): 
        raise NotImplementedError

class DecisionNode(TreeNodeInterface):
    """Class representing an internal node of a decision tree."""

    def __init__(self, test_attr_name, child_0, child_1, child_2, child_3, child_4, split):
        """Constructor for the decision node.               
        """    
        self.attribute = test_attr_name  
        self.child_0 = child_0
        self.child_1 = child_1
        self.child_2 = child_2
        self.child_3 = child_3
        self.child_4 = child_4
        self.split = split
    

    def classify(self, example): 
        """Classify an example based on its test attribute value.
        
        Args:
            example: representing a data instance

        Returns: a class label
        """
        test_val = example[self.attribute]
        lst = df[self.attribute].value_counts().keys() #possible values of the attribute
        if dataTypeDict[self.attribute] == 'int64' and len(lst) <= 5:
            if test_val == 0:
                return self.child_0.classify(example)
            elif test_val == 1:
                return self.child_1.classify(example)
            elif test_val == 2:
                return self.child_2.classify(example)
            elif test_val == 3:
                return self.child_3.classify(example)
            elif test_val == 4:
                return self.child_4.classify(example)
        else:
            if test_val <= self.split:
                return self.child_0.classify(example)
            else:
                return self.child_1.classify(example)

class LeafNode(TreeNodeInterface):
    """Class representing a leaf node of a decision tree.  Holds the LABEL."""

    def __init__(self, label):
        """Constructor for the leaf node.

        Args:
            label: class label for the majority class that this leaf represents
        """    
        self.label = label

    def classify(self, example):
        """Classify an example.
        
        Args:
            example: representing a data instance

        Returns: a class label
        """
        return self.label

#Calculate Entropy
def get_entropy(y_data):
    if 0 in y_data.value_counts().keys():
        prob_0 = y_data.value_counts()[0] / len(y_data)
        entropy_0 = prob_0 * log2(prob_0)
    else:
        prob_0 = 0
        entropy_0 = 0
    
    if 1 in y_data.value_counts().keys():
        prob_1 = y_data.value_counts()[1] / len(y_data)
        entropy_1 = prob_1 * log2(prob_1)
    else:
        prob_1 = 0
        entropy_1 = 0
    return -(entropy_0 + entropy_1)

    
#Calculate the max infogain
def maxInfoGain(attributes, X_data, y_data):
    best_attribute = ""
    maxInfo = float("-inf")
    for attribute in attributes:
        p_entropy = get_entropy(y_data)
        c_entropy = 0
        lst = df[attribute].value_counts().keys() #possible values of the attribute
        if dataTypeDict[attribute] == 'int64' and len(lst) <= 5:
            val_counts = X_data[attribute].value_counts()
            for val in val_counts.keys():
                idx = X_data.index[X_data[attribute] == val].tolist()
                ch_data = y_data[idx]
                c_entropy += (val_counts[val] / len(y_data)) * get_entropy(ch_data)
        else:
            val_counts = X_data[attribute].value_counts().keys()
            split = mean(val_counts)
            idx = X_data.index[X_data[attribute] <= split].tolist()
            ch_data1 = y_data[idx]
            idx = X_data.index[X_data[attribute] > split].tolist()
            ch_data2 = y_data[idx]
            c_entropy = (len(ch_data1)/len(y_data)) * get_entropy(ch_data1) + (len(ch_data2)/len(y_data)) * get_entropy(ch_data2)
        
        infoGain = p_entropy - c_entropy
        if maxInfo < infoGain:
            maxInfo = infoGain
            best_attribute = attribute
    return best_attribute


#Train a Decision tree
def decision_tree(X_data, y_data, attributes):
    if len(X_data) == 0:
        return None

    if len(attributes) == 0:
        return LeafNode(y_data.value_counts().keys()[0])
    
    if(len(X_data) < 10): #Stopping criterion
        return LeafNode(y_data.value_counts().keys()[0])

    rand_att = random.sample(attributes, math.floor(math.sqrt(len(attributes))))
    attribute = maxInfoGain(rand_att, X_data, y_data)
    val_lst = df[attribute].value_counts().keys() #possible values of the attribute
    curr_values = X_data[attribute].value_counts() #current values of the attribute in the arbitrary dataset X_data
    child_0 = child_1 = child_2 = child_3 = child_4 = None
    if dataTypeDict[attribute] == 'int64' and len(val_lst) <= 5:
        for val in val_lst:
            if val not in curr_values:
                return LeafNode(y_data.value_counts().keys()[0])
            if val == 0:
                idx_lst = X_data.index[X_data[attribute] == 0].tolist()
                child_0 = decision_tree(X_data[X_data[attribute] == 0], y_data[idx_lst], attributes)
            if val == 1:
                idx_lst = X_data.index[X_data[attribute] == 1].tolist()
                child_1 = decision_tree(X_data[X_data[attribute] == 1], y_data[idx_lst], attributes)
            if val == 2:
                idx_lst = X_data.index[X_data[attribute] == 2].tolist()
                child_2 = decision_tree(X_data[X_data[attribute] == 2], y_data[idx_lst], attributes)
            if val == 3:
                idx_lst = X_data.index[X_data[attribute] == 3].tolist()
                child_3 = decision_tree(X_data[X_data[attribute] == 3], y_data[idx_lst], attributes)
            if val == 4:
                idx_lst = X_data.index[X_data[attribute] == 4].tolist()
                child_4 = decision_tree(X_data[X_data[attribute] == 4], y_data[idx_lst], attributes)
        return DecisionNode(attribute, child_0, child_1, child_2, child_3, child_4, 0)
    else:
        curr_values = X_data[attribute].value_counts().keys()
        split = mean(curr_values)
        idx_lst1 = X_data.index[X_data[attribute] <= split].tolist()
        child_0 = decision_tree(X_data[X_data[attribute] <= split], y_data[idx_lst1], attributes)
        idx_lst2 = X_data.index[X_data[attribute] > split].tolist()
        child_1 = decision_tree(X_data[X_data[attribute] > split], y_data[idx_lst2], attributes)
        return DecisionNode(attribute, child_0, child_1, None, None, None, split=split)


#Load the dataset

#TO LOAD THE HOUSE_VOTES DATASET UNCOMMENT THE NEXT LINE
#df = pd.read_csv ('datasets\hw3_house_votes_84.csv')

#TO LOAD THE CONTRACEPTIVES DATASETS UNCOMMENT THE NEXT LINE
df = pd.read_csv ('datasets\cmc.data')

#TO LOAD THE WINE DATASET UNCOMMENT THE NEXT 2 LINES
# df = pd.read_csv ('datasets\hw3_wine.csv', sep='\t')
# df = df.rename(columns={'# class': 'class'})

dataTypeDict = dict(df.dtypes)

#Split based on class label
print("Splitting datasets")
datasets = {}
split_dataset = df.groupby(df.loc[:,'class'])
for groups, data in split_dataset:
    datasets[groups] = data

#Create stratified k-folds
k = 10
kfolds = {}
for i in range(1,k+1):
    kfolds[i] = pd.DataFrame()
    for key in datasets:
        samples = datasets[key].sample(len(split_dataset.get_group(key))//k, replace=False)
        datasets[key] = datasets[key].drop(samples.index)
        kfolds[i] = pd.concat([kfolds[i], samples])

#Random Forest Training & Testing
metrics = {}
ntrees = [1,5,10,20,30,40,50]
for val in ntrees:
    metrics[val] = {'acc': 0, 'prec': 0, 'rec': 0, 'f': 0}
for key in kfolds:
    print("For k = ", key)
    test_data = kfolds[key]
    train_data = pd.DataFrame()
    for key1 in kfolds:
        if(key != key1):
            train_data = pd.concat([train_data, kfolds[key1]])

    x_test = test_data.loc[:, test_data.columns != 'class']
    y_test = test_data.loc[:, 'class']
    for ntree in ntrees:
        print("Random Forest for ntree = ", ntree)
        forest = []
        correct = {}
        incorrect = {}
        for val in df.loc[:, 'class'].value_counts().keys():
            correct[val] = 0
            d = {}
            incorrect[val] = d

        for i in range(ntree):
            bootstrap_data = train_data.sample(len(train_data), replace=True) 
            x_train = bootstrap_data.loc[:, bootstrap_data.columns != 'class']
            x_train = x_train.reset_index(drop=True)
            y_train = bootstrap_data.loc[:, 'class']
            y_train = y_train.reset_index(drop=True)
            column_lst = [col for col in x_train]
            root = decision_tree(x_train, y_train, column_lst)
            forest.append(root)
        for index, row in x_test.iterrows():
            pred = []
            for tree in forest:
                val = tree.classify(row)
                pred.append(val)
            y = max(pred, key=pred.count)
            if y == y_test[index]:
                correct[y] = correct[y] + 1
            else:
                if y in incorrect[y_test[index]]:
                    incorrect[y_test[index]][y] = incorrect[y_test[index]][y] + 1
                else:
                    incorrect[y_test[index]][y] = 1
        acc = 0
        prec = 0
        rec = 0
        f = 0
        for i in correct:
            acc += correct[i]
            sum_p = correct[i]
            sum_r = correct[i]
            for j in incorrect:
                if j == i:
                    for key1 in incorrect[j]:
                        sum_r += incorrect[j][key1]
                else:
                    if i in incorrect[j]:
                        sum_p += incorrect[j][i]
            if sum_p != 0:
                prec += correct[i]/sum_p
            if sum_r != 0:
                rec += correct[i]/sum_r
        acc = acc/len(x_test)
        prec = prec/len(correct)
        rec = rec/len(correct)
        f = prec*rec/(prec+rec)
        metrics[ntree]['acc'] = metrics[ntree]['acc'] + acc
        metrics[ntree]['prec'] = metrics[ntree]['prec'] + prec
        metrics[ntree]['rec'] = metrics[ntree]['rec'] + rec
        metrics[ntree]['f'] = metrics[ntree]['f'] + f

acc_lst = []
prec_lst = []
rec_lst = []
f_lst = []
for ntree in metrics:
    metrics[ntree]['acc'] = metrics[ntree]['acc']/k
    acc_lst.append(metrics[ntree]['acc'])
    metrics[ntree]['prec'] = metrics[ntree]['prec']/k
    prec_lst.append(metrics[ntree]['prec'])
    metrics[ntree]['rec'] = metrics[ntree]['rec']/k
    rec_lst.append(metrics[ntree]['rec'])
    metrics[ntree]['f'] = metrics[ntree]['f']/k
    f_lst.append(metrics[ntree]['f'])
    print(metrics[ntree])

plt.scatter(ntrees, acc_lst)
plt.xlabel("nTrees")
plt.ylabel("Accuracy of testing data")
plt.show()

plt.scatter(ntrees, prec_lst)
plt.xlabel("nTrees")
plt.ylabel("Precision of testing data")
plt.show()

plt.scatter(ntrees, rec_lst)
plt.xlabel("nTrees")
plt.ylabel("Recall of testing data")
plt.show()

plt.scatter(ntrees, f_lst)
plt.xlabel("nTrees")
plt.ylabel("F-Score of testing data")
plt.show()
