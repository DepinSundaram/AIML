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


class ExplainatoryDataAnalysis:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
    
    def frame(self):
        return self.df
    
    def info(self):
        self.df.info()
        
    def head(self, n):
        return self.df.head(n)
    
    def value_counts(self, columns):
        for col in columns:
            print("Value Counts for ", col)
            print(self.df[col].value_counts())
            print("\n")
            
    def drop(self, columns, inplace=False):
        self.df.drop(columns=columns, inplace=inplace)
        print("Dropped the columns!")
    
    def encode(self, columns, columns_renaming, encode_map):
        index = 0
        for col in columns:
            self.df[columns_renaming[index]] = self.df[col].apply(lambda x: encode_map[x])
            if col!=columns_renaming[index]:
                self.df.drop(columns=[col], inplace=True)
            index = index+1
        print("Encoding done!")
        
    def fill_underscore(self, columns):
        for col in columns:
            self.df[col] = self.df[col].apply(lambda x: re.sub("[\s\-]+", "_", x))
            self.df[col] = self.df[col].apply(lambda x: re.sub("[\(\)]+", "", x))
            self.df[col] = self.df[col].apply(lambda x: re.sub("[\<]+", "lt", x))
            self.df[col] = self.df[col].apply(lambda x: re.sub("[\&]+", "", x))
            self.df[col] = self.df[col].apply(lambda x: re.sub("[\~]+", "_", x))
            print("Space replaced with underscore for column and braces removed", col)
        print("Space replaced with underscore for all columns")
    
    def get_dummies(self, columns, drop_columns):
        category_dummies = pd.get_dummies(self.df[columns])
        self.df.drop(columns=drop_columns, inplace=True)
        self.df = pd.concat([self.df, category_dummies], axis=1)
        print("1. Created dummies for the columns", columns)
        print("2. Dropped columns", drop_columns)
        print("3. Updated data frame with dummy columns")
        return self.head(10)
   
    def corr(self, reference_column, ascending):
        return self.df.corr()[reference_column].sort_values(ascending=ascending)
    
    def select(self, columns):
        return self.df[columns]
    
    def heatmap(self, columns):
        df_selected = pd.DataFrame()
        if len(columns)>0:
            df_selected = self.df[columns]
        else:
            df_selected = self.df
            
        fig, ax = plt.subplots(figsize=(25,10))
        sns.heatmap(df_selected.corr(), cmap="YlGnBu", annot=False, fmt = '.4g', cbar=False, ax=ax)
        