#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve,auc


class Modals:
    def __init__(self, dataframe):
        self.df = dataframe
        
    def train_test_split(self, target_column, test_size, random_state):
        self.X = self.df.drop(columns=[target_column]).copy()
        self.y = self.df[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        
    def get_best_parameters_dt(self, params_grid):
        clf = DecisionTreeClassifier(random_state=80)
        cv_model = GridSearchCV(clf, param_grid = params_grid)
        cv_model.fit(self.X_train, self.y_train)
        print(cv_model.best_params_)
        self.dtree_best_params = cv_model.best_params_
        return cv_model.best_params_
        
    def dtree(self, depths):
        ascore = []
        dectreearr = []
        for depth in range(1, depths):
            dectree = DecisionTreeClassifier(max_depth=depth, criterion='gini')
            dectree_train = dectree.fit(self.X_train, self.y_train)
            y_pred = dectree_train.predict(self.X_test)
            acc_score = accuracy_score(self.y_test,y_pred)*100
            ascore.append(acc_score)
            dectreearr.append(dectree_train)
            print("#", depth, "Decision tree accuracy Score=",accuracy_score(self.y_test,y_pred)*100)
        self.dtree_scores = {'ascore':ascore, 'dectree':dectreearr}
        return {'ascore':ascore, 'dectree':dectreearr}
    
    def scale(self, arr):
        arr_scaled = np.array(arr)-np.array(arr).min()
        return arr_scaled
    
    def bar(self, arr) :
        fig, ax = plt.subplots(figsize=(15,10))
        ax.bar(range(1,len(arr)+1), arr)
        return arr
    
    def decision_tree_classification(self, max_depth, criterion):
        dtc = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
        dtc = dtc.fit(self.X_train, self.y_train)
        self.y_pred = dtc.predict(self.X_test)
        self.accurecy_score_pt = accuracy_score(self.y_test,self.y_pred)*100
        self.cm = confusion_matrix(self.y_test,self.y_pred)
        self.probability = dtc.predict_proba(self.X_test)
        self.positive_probability = self.probability[:,1]
        print("Accurecy Score is:", self.accurecy_score_pt)
        print("Following is the Confusion Matrix.")
        print("-------------------------------------")
        fig, ax = plt.subplots(figsize=(5,3))
        sns.heatmap(self.cm, cmap="YlGnBu", annot=True, fmt = '.4g', cbar=False, ax=ax)
    
    def measurements(self):
        TN = self.cm[0,0] #Actually 0, but predicted as 0: TN
        FP = self.cm[0,1] #Actually 0, but predicted as 1 : FP
        FN = self.cm[1,0] #Actually 1, but predicted as 0 : FN
        TP = self.cm[1,1] #Actually 1, but predicted as 1 : TP
        N = len(self.y_test)
        
        accuracy = (TN+TP)/N
        recall = TP/(TP+FN)
        specificity = TN/(TN+FP)
        fpr = 1-specificity
        precision = TP/(TP+FP)
        f1_score_value = f1_score(self.y_test, self.y_pred)
        frame_contents = {"Measurement":["Accuracy", "Recall or TPR or Sensitivity", "Specificity", "FPR or 1-Specificity", "Precision", "F1 Score"], "Values": [accuracy, recall, specificity, fpr, precision, f1_score_value]}
        df_measurements = pd.DataFrame(frame_contents)
        return {"measurements":df_measurements}
        