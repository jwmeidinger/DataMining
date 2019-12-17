## Authors: Jayden Rosenau, Jordan Meidinger
## Final Project Data Mining: UFC Classification
## Data: 12/12/2019

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import RandomOverSampler


## Read data
df = pd.read_csv('data.csv')
pd.set_option('display.max_rows', 200)

## Print Nulls
""" for column in df.columns:
    if df[column].isnull().sum()!=0:
        print(f"Nan in {column}: {df[column].isnull().sum()}") """


## Dropping items
drop_na = df[df['B_avg_opp_TD_att'].isnull()].index
df2 = df.drop(drop_na, axis=0)
drop_na = df2[df2['R_avg_opp_TD_att'].isnull()].index
df2 = df2.drop(drop_na, axis=0)

""" for column in df2.columns:
    if df2[column].isnull().sum()!=0:
        print(f"Nan in {column}: {df2[column].isnull().sum()}")   
"""
df2.drop(columns=['Referee'], inplace = True)

plt.scatter('R_Height_cms', 'R_Reach_cms',data = df2)
##plt.show()

df2['R_Reach_cms'].fillna(df2['R_Height_cms'], inplace=True)
df2['B_Reach_cms'].fillna(df2['B_Height_cms'], inplace=True)
df2.fillna(df2.median(), inplace=True)

df2.drop(df2.index[df2['Winner'] == 'Draw'], inplace = True)
df2.drop(columns=['location', 'date', 'R_fighter', 'B_fighter'], inplace=True)


## taking weight classes and making them boolean
df2 = pd.concat([df2, pd.get_dummies(df2[['weight_class', 'B_Stance', 'R_Stance']])], axis=1)
df2.drop(columns=['weight_class', 'B_Stance', 'R_Stance'], inplace=True)


##print(df2.T)

df_num = df.select_dtypes(include=[np.float, np.int])
scaler = StandardScaler()
df[list(df_num.columns)] = scaler.fit_transform(df[list(df_num.columns)])

########DATA split
y = df2['Winner']
X = df2.drop(columns = 'Winner')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=40)

##############
## RandomForest
##############

model = RandomForestClassifier(n_estimators= 100, oob_score=True, random_state=43)
model.fit(X_train, y_train)

##print(model.oob_score_)


##print(model.score(X_test,y_test))

## Show important features
""" feat_imps = {}
for i, imp in enumerate(model.feature_importances_):
    feat_imps[X_train.columns[i]] = imp

sorted_imp_feats = (sorted(feat_imps.items(), key = lambda x: x[1], reverse=True))
print(sorted_imp_feats)  """


###########################
##RandomForest OverSampling
###########################


ros = RandomOverSampler(random_state = 0)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

model2 = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=43)
model2.fit(X_resampled, y_resampled)
score2 = 0
""" for x in range(5):
    y_predic2 = model2.predict(X_test)
    score2 += accuracy_score(y_test, y_predic2)
print(score2 /5)
print(model2.oob_score_) """


######################
## AdaBoost Classifier
######################
clf = AdaBoostClassifier(RandomForestClassifier(),n_estimators=100 ,learning_rate= .5)
clf.fit(X_train, y_train)
feat_imps1 = {}
""" for i, imp in enumerate(clf.feature_importances_):
    feat_imps1[X_train.columns[i]] = imp

sorted_imp_feats1 = (sorted(feat_imps1.items(), key = lambda x: x[1], reverse=True))
print(sorted_imp_feats1) 
result = 0 """
""" for x in range(5):
    result += clf.score(X_test,y_test) 
print(result/5) """


##############
## Naive Bayes
##############
gaussian = GaussianNB()
gaussian.fit(X_train,y_train)
result = gaussian.score(X_test,y_test)
##print(result)


######################################
##Classification with only top features 
######################################
df3 = (df2[['Winner','R_avg_opp_SIG_STR_landed',
'R_age',
'B_avg_HEAD_att',
'R_avg_opp_SIG_STR_pct',
'R_avg_opp_HEAD_landed',
'B_avg_DISTANCE_att',
'R_avg_opp_DISTANCE_landed',
'B_avg_DISTANCE_landed',
'R_avg_opp_DISTANCE_att',
'R_avg_opp_TOTAL_STR_landed',
'R_avg_BODY_att',
'B_avg_opp_DISTANCE_att',
'B_avg_SIG_STR_landed',
'B_avg_opp_SIG_STR_pct',
'B_avg_BODY_landed',
'R_avg_opp_LEG_landed']])

y1 = df3['Winner']
X1 = df3.drop(columns = 'Winner')

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.10, random_state=40)

model2 = RandomForestClassifier(n_estimators= 100, oob_score=True, random_state=43)
model2.fit(X_train1, y_train1)
result3 = 0
for x in range(5):
    result3 += model2.score(X_test1,y_test1)

#print(model2.oob_score_)
#print(result3/5)

clf1 = AdaBoostClassifier(RandomForestClassifier(),n_estimators=1000 ,learning_rate= 1)
clf1.fit(X_train1, y_train1)


result4 = 0 
for x in range(5):
    result4 += clf1.score(X_test1,y_test1) 
#print(result4/5) 
