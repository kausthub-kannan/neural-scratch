from sklearn.datasets import load_breast_cancer
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.model_selection import train_test_split
 from sklearn import metrics
 import pandas as pd
 import numpy as np
 from matplotlib import pyplot as plt
 import seaborn as sns
 sns.set_style('darkgrid') 

 # choose a binary classification problem
 data = load_breast_cancer()
 # develop predictors X and target y dataframes
 X = pd.DataFrame(data['data'], columns=data['feature_names'])
 y = abs(pd.Series(data['target'])-1)
 # split data into train and test set in 80:20 ratio
 X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)
 # build a RF model with default parameters
 model = RandomForestClassifier(random_state=1)
 model.fit(X_train, y_train)
 preds = model.predict(X_test)
