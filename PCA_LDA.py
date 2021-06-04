# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:32:22 2019

@author: milk
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import datasets  
#1
wine = datasets.load_wine()
data = pd.DataFrame(wine.data,columns=wine.feature_names)
label = pd.DataFrame(wine.target,columns=['wine_class'])
wine_features = data.iloc[:,0:-1]
x_train,x_test,y_train,y_test=train_test_split(data,label,
                                               test_size = 0.3,
                                               random_state = 0)
forest = RandomForestClassifier(n_estimators = 10000,
                                random_state = 0,n_jobs = -1)
forest.fit(x_train, y_train)
rf_features = wine_features.columns
importances = forest.feature_importances_
indices = np.argsort(importances)

for i in range(rf_features.shape[0]):
    plt.bar(i,importances[indices[i]],color = 'pink',edgecolor = 'k')
    plt.xticks(np.arange(rf_features.shape[0]),rf_features,
               rotation = 90,fontsize = 15)
plt.title('feature selection')
plt.ylabel('relative importance')

#2(PCA)
X=wine.data
Y=wine.target
pca_wine=PCA().fit(X)
plt.figure()
print(pca_wine.explained_variance_ratio_)
plt.plot(np.cumsum(pca_wine.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

pca_wine=PCA(n_components=2)
pca_wine_feature=pca_wine.fit_transform(X)
plt.figure()
plt.title('PCA(component=2)')
colors = ['r', 'b', 'g']
for l, c in zip(np.unique(Y), colors):
    plt.scatter(pca_wine_feature[Y==l, 0], pca_wine_feature[Y==l, 1], c=c, label=l)
plt.legend(loc='upper right')

#2(LDA)
lda=LinearDiscriminantAnalysis()
lda_wine=lda.fit(X,Y)
print(lda.explained_variance_ratio_)

lda=LinearDiscriminantAnalysis(n_components=2)
lda_wine=lda.fit_transform(X,Y)
plt.figure()
plt.title('LDA(component=2)')
colors = ['r', 'b', 'g']
for l, c in zip(np.unique(Y), colors):
    plt.scatter(lda_wine[Y==l, 0], lda_wine[Y==l, 1], c=c, label=l)
plt.legend(loc='upper right')

#3
airstraw=pd.read_csv("C:\\Users\\milk3\\Documents\\Machine Learning\\HW4\\Airstraw.csv")
airstraw["sponsor_ID"]=airstraw["贊助人ID"].fillna(airstraw["贊助人ID"].mean())
airstraw["postal_code"]=airstraw["郵遞區號"].fillna("300")
airstraw["country"]=airstraw["國家"].fillna("台灣（本島）")
airstraw["sponorship_option_ID"]=airstraw["贊助選項 ID"].fillna("雙人NanaQ")
airstraw["price"]=airstraw["選項價錢"].fillna(airstraw["選項價錢"].mean())
airstraw["total_price"]=airstraw["總金額"].fillna(airstraw["總金額"].mean())
airstraw["pay"]=airstraw["付款方式"].fillna("信用卡付款")
airstraw["shopping_guide"]=airstraw["導購(分析項目)"]
features=airstraw.loc[:,["sponsor_ID","postal_code","country","sponorship_option_ID",
                         "price","total_price","pay"]]
le=preprocessing.LabelEncoder()
le.fit(features["country"])
features["country"]=le.transform(features["country"])
le.fit(features["sponorship_option_ID"])
features["sponorship_option_ID"]=le.transform(features["sponorship_option_ID"])
le.fit(features["pay"])
features["pay"]=le.transform(features["pay"])
le.fit(airstraw["shopping_guide"])
airstraw["shopping_guide"]=le.transform(airstraw["shopping_guide"])

for x in ["sponsor_ID","postal_code","price","total_price"]:
    features[x]=le.fit_transform(features[x])

x_train,x_test,y_train,y_test=train_test_split(features,airstraw.shopping_guide,
                                               test_size = 0.3,
                                               random_state = 0)
forest = RandomForestClassifier(n_estimators = 10000,
                                random_state = 0,n_jobs = -1)
forest.fit(x_train, y_train)
rf_features = features.columns
importances = forest.feature_importances_
indices = np.argsort(importances)

for i in range(rf_features.shape[0]):
    plt.bar(i,importances[indices[i]],color = 'pink',edgecolor = 'k')
    plt.xticks(np.arange(rf_features.shape[0]),rf_features,
               rotation = 90,fontsize = 15)
plt.title('feature selection')
plt.ylabel('relative importance')
'''
random_forest=RandomForestRegressor()
random_forest.fit(features,airstraw.shopping_guide)

rf_features=features.columns
importances=random_forest.feature_importances_
indices=np.argsort(importances)
plt.figure()
plt.title('feature selection')
plt.barh(range(len(indices)),importances[indices],color='pink',align='center')
plt.yticks(range(len(indices)),[rf_features[i] for i in indices])
plt.xlabel('Relative importances of indicators')
plt.show()