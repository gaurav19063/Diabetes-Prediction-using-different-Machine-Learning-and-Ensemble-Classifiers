import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score,matthews_corrcoef
import numpy as np
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import argparse # Load argparser to read input from the user
from argparse import RawTextHelpFormatter # Import RawTextHelpFormatter used to print helper message
# ======================= Cross validation for performance analysis.===========================
def CV(model,X,y):
  scoring = {'acc': 'accuracy',
           'AUC':'roc_auc',
            'f1':'f1'
           }
  scores = cross_validate(model, X, y, scoring=scoring,
                          cv=5, return_train_score=True)
  score_df=pd.DataFrame.from_dict(scores)
  print("\nScores for 5 folds:")
  print(score_df)
  return

#============================== Decision Tree Classifier===========================================================

from sklearn import tree
from sklearn.metrics import accuracy_score
def DT(X_train,y_train,X_test):
  clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
  y_pred=clf.predict(X_test)
  return y_pred

# ================================ XG boost ===========================================
import xgboost as xgb
from sklearn.metrics import mean_squared_error
def XG(X_train,y_train,X_test):

  parameters = {
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.5, 1.0],
        'max_depth': [3, 4, 5]
        }
  model = GridSearchCV(xgb.XGBClassifier(objective = "binary:logistic", eval_metric = 'error'),
                    parameters,
                    )
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  return y_pred,model
# ================================ Random Forest ===========================================

from sklearn.ensemble import RandomForestClassifier
def RF(X_train,y_train,X_test):
  param_grid = {
    'n_estimators': [100,200, 300,400],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [10, 20, 30, 40, 50],
    'criterion' :['gini', 'entropy']
  }
  rfc=RandomForestClassifier(random_state=42)
  clf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
  clf.fit(X_train,y_train)
  y_pred=clf.predict(X_test)
  return y_pred,clf
# ================================ Naive Bayes ===========================================

from sklearn.naive_bayes import GaussianNB
def GNB(X_train,y_train,X_test):
  params = {'var_smoothing': [0.01, 0.1, 0.5, 1.0],
         }
  rfc = GaussianNB()
  clf = GridSearchCV(estimator=rfc, param_grid=params, cv= 5)
  clf.fit(X_train,y_train)
  y_pred=clf.predict(X_test)
  return y_pred,clf
# ================================ Adaboost ===========================================
from sklearn.ensemble import AdaBoostClassifier
def ADB(X_train,y_train,X_test):

  params= {
              "n_estimators": [100,200, 300,400],
              "learning_rate": [0.01, 0.1, 0.5]
             }
  rfc= AdaBoostClassifier()
  clf = GridSearchCV(estimator=rfc, param_grid=params, cv= 5)
  clf.fit(X_train,y_train)
  y_pred=clf.predict(X_test)
  return y_pred,clf

from sklearn.ensemble import BaggingClassifier
def bagging(X_train,y_train,X_test):
  params={
      "n_estimators": [100,200, 300,400]
  }
  rfc= BaggingClassifier()
  clf = GridSearchCV(estimator=rfc, param_grid=params, cv= 5)
  clf.fit(X_train,y_train)
  y_pred=clf.predict(X_test)
  return y_pred,clf
def BestHyperParams_GridSearchCV(model,param_grid,X_train,Y_train):
  # print("Grid Search for hyper parameter tunning.")
  grid = GridSearchCV(model,param_grid,refit = True, verbose=2,cv=5, scoring='accuracy')
  grid.fit(X_train,Y_train)
  return grid.best_params_

def main(path):
  data=pd.read_csv(path)
  # =========================== Finding null value =======================================
  # data.fillna(data.mean())
  print("Is there any Null value? ",data.isnull().any().any())
  # =========================== plotting class distribution =======================================
  sns.countplot(data['Outcome'])
  plt.show()
  # =========================== Cheacking Corrilation Between the different Features =======================================
  # sns.heatmap(data,annot=True, fmt="g", cmap='viridis')
  corr = data.corr()
  sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns,annot=True)
  plt.show()
  X, Y =data.drop(['Outcome'],axis=1),data['Outcome']
  # oversample = RandomOverSampler(sampling_strategy='minority',random_state=48)
  # # fit and apply the transform
  # X_over, y_over = oversample.fit_resample(X, Y)
  # summarize class distribution
  # print(Counter(y_over))

  param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}
  model=SVC()
  # print("Best hyperparameters",BestHyperParams_GridSearchCV(model,param_grid,X,Y))
  # ==========================spliting data into 70:30 train test data ==============================
  x_train, x_test, y_train, y_test = train_test_split(X, Y ,test_size=0.3, random_state=42)
  #============================== MultiLayerPerceptron Classifier===========================================================
  from sklearn.neural_network import MLPClassifier
  clf = MLPClassifier(random_state=1, solver="sgd",max_iter=700).fit(x_train, y_train)
  pred_MLP=clf.predict(x_test)
  print(classification_report(y_test, pred_MLP, target_names=["0","1"]))
  # =========== Evaluating ML models = KNN,DT and deep learning model = MLP ,ensemble model XGboosting on 30 % test data ============


  print("========================= For Decision Tree")
  CV(tree.DecisionTreeClassifier(),x_train,y_train)
  pred_DT=DT(x_train,y_train,x_test)
  print(classification_report(y_test, pred_DT, target_names=["0","1"]))

  print("========================== For XG boosting")
  pred_XG,model=XG(x_train,y_train,x_test)
  CV(model,x_train,y_train)
  print(classification_report(y_test, pred_XG, target_names=["0","1"]))

  print("=========================== For MLP")
  CV(clf,x_train,y_train)
  print(classification_report(y_test, pred_MLP, target_names=["0","1"]))
  # =========== Evaluating ML models = Random Forest,Naive Bayes , Adaboost on 30 % test data ============

  print("======================== For Adaboost")
  pred_ADB,model=ADB(x_train,y_train,x_test)
  CV(model,x_train,y_train)
  print(classification_report(y_test, pred_ADB, target_names=["0","1"]))
  print("======================== For Naive Bayes")
  pred_NB,model=GNB(x_train,y_train,x_test)
  CV(model,x_train,y_train)
  print(classification_report(y_test, pred_NB, target_names=["0","1"]))
  print("======================== For Bagging")
  pred_bag,model=bagging(x_train,y_train,x_test)
  CV(model,x_train,y_train)
  print(classification_report(y_test, pred_bag, target_names=["0","1"]))

# command line input from the user
parser = argparse.ArgumentParser(description='Please provide following arguments',formatter_class=RawTextHelpFormatter)
parser.add_argument("-i","-I","--input", type=str, required=True, help="Input address of train file")
args = parser.parse_args() # Take command line input and store in the args object

path_train= args.input #train file address



main(path_train)
