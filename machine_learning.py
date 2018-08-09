
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy
import re
import json
import jieba
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from ipywidgets import interact
from mpl_toolkits.mplot3d import Axes3D
from gensim.models import word2vec

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.datasets import load_digits
from sklearn.datasets import load_iris

import warnings
warnings.filterwarnings('ignore')


# In[2]:


from matplotlib.font_manager import FontProperties

font = FontProperties(family='sans-serif', 
                      fname='/Users/johnnie/Library/Fonts/Microsoft YaHei.ttf', 
                      style='italic', weight='bold', size='large')


# In[3]:


from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from tpot import TPOTClassifier


# In[4]:


from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier


# In[5]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from tpot import TPOTRegressor


# In[6]:


from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor


# In[7]:


from sklearn.cluster import KMeans, SpectralClustering


# In[8]:


from sklearn.decomposition import PCA, NMF, LatentDirichletAllocation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS


# In[9]:


from apriori import apriori, generateRules
from fp_growth import find_frequent_itemsets


# In[10]:


from hmmlearn.hmm import GaussianHMM, MultinomialHMM, GMMHMM
from hmmlearn.base import _BaseHMM


# In[11]:


from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures, Imputer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[12]:


from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel


# In[13]:


from sklearn.model_selection import GridSearchCV, train_test_split


# In[14]:


from sklearn.model_selection import learning_curve


# In[15]:


from sklearn.externals import joblib


# In[16]:


# 数据集切分（训练集和验证集）
def split_train_test(data, test_size=0.3):
    split_train, split_val = train_test_split(data, test_size=test_size, random_state=0)
    return split_train, split_val


# In[17]:


# 数据下采样
def downsampling(data, ratio, classes):
    n_classes = []
    for i in range(len(classes)):
        n_classes.append(len(data[data.columns[-1]] == classes[i]))
    if np.max(n_classes) >= ratio * np.min(n_classes):
        resampling = resample(data[data[data.columns[-1]] == classes[find(n_classes.find(np.max(n_classes)))]],
                                              replace=False, n_samples=np.min(n_classes) * ratio, random_state=0)
        return resampling
    else:
        print('Balanced samples, no need to downsampling any more!')
        return n_classes


# In[18]:


# 特征选择
def featuresselection(X, y, k=2, n_features_to_select=2, estimator=None,
                      selectkbest=False, rfe=False, selectfrommodel=False):
    if selectkbest:
        return SelectKBest(k=2).fit_transform(X, y)
    if rfe:
        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
        return rfe.fit_transform(X, y)
    if selectfrommodel:
        model = SelectFromModel(estimator=estimator, prefit=True)
        return model.transform(X)


# In[19]:


# 网格搜索参数
def gscv(estimator, param_grid, cv, X, y, classifier=False, regressor=False, cluster=False, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
    estimator.fit(X, y, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
    X = estimator.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
    if classifier:
        gscv = GridSearchCV(estimator=estimator.classifier, param_grid=param_grid, cv=cv)
    elif regressor:
        gscv = GridSearchCV(estimator=estimator.regressor, param_grid=param_grid, cv=cv)
    elif cluster:
        gscv = GridSearchCV(estimator=estimator.cluster, param_grid=param_grid, cv=cv)
    gscv.fit(X, y)
    return gscv.best_score_, gscv.best_params_


# In[20]:


# 机器学习模型整合
def MachineLearning(X_train=None, y_train=None, X_test=None, y_test=None, transactions=None, threshold=0.3, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False,
                   ngram_range=(1, 1), max_features=None, early_stopping_rounds=None, eval_metric=None, eval_set=None,
                   classification=False, logisticregression=False, decisiontreeclassifier=False, svc=False, kneighborsclassifier=False, probability=False, bernoullinb=False, 
                   multinomialnb=False, gaussiannb=False, basehmm=False, multinomialhmm=False, gaussianhmm=False, gmmhmm=False, baggingclassifier=False, randomforestclassifier=False, 
                   adaboostclassifier=False, gradientboostingclassifier=False, xgbclassifier=False, lgbmclassifier=False, mlpclassifier=False, tpotclassifier=False,
                   regression=False, linearregression=False, decisiontreeregressor=False, svr=False, kneighborsregressor=False, baggingregressor=False,
                   randomforestregressor=False, adaboostregressor=False, gradientboostingregressor=False, xgbregressor=False, lgbmregressor=False, mlpregressor=False, tpotregressor=False,
                   clustering=False, kmeans=False, spectralclustering=False, decomposition=False, pca=False, nmf=False, isomap=False, lda=False, association=False, aprior=False, fp_growth=False):
    if classification:
        if logisticregression:
            LRClassifier = LogisticR(ngram_range=ngram_range, max_features=max_features)
            bestscore, bestparams = gscv(LRClassifier, {'C': [0.01, 0.1, 1.0, 10.0]}, 5, X_train, y_train,
                classifier=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            LRClassifier = LogisticR(ngram_range=ngram_range, max_features=max_features, classifier=LogisticRegression(C=bestparams['C'], n_jobs=-1))
            LRClassifier.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = LRClassifier.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            pred_probs = LRClassifier.predict_proba(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():
                score = LRClassifier.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return LRClassifier, preds, pred_probs, score
            else:
                return LRClassifier, preds, pred_probs
        if decisiontreeclassifier:
            DTClassifier = DecisionTreeC(ngram_range=ngram_range, max_features=max_features)
            bestscore, bestparams = gscv(DTClassifier, {'max_depth': [5, 8, 15, 25],
                                                  'min_samples_split': [2, 5, 10, 15],
                                                  'min_samples_leaf': [1, 2, 5, 10]}, 5, X_train, y_train,
                                                  classifier=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            DTClassifier = DecisionTreeC(ngram_range=ngram_range, max_features=max_features, classifier=DecisionTreeClassifier(max_depth=bestparams['max_depth'], min_samples_split=bestparams['min_samples_split'], 
                                                       min_samples_leaf=bestparams['min_samples_leaf']))
            DTClassifier.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = DTClassifier.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            pred_probs = DTClassifier.predict_proba(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():
                score = DTClassifier.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return DTClassifier, preds, pred_probs, score
            else:
                return DTClassifier, preds, pred_probs
        if svc:
            SVClassifier = SVClassfication(ngram_range=ngram_range, max_features=max_features)
            bestscore, bestparams = gscv(SVClassifier, {'C': [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}, 5, X_train, y_train,
                classifier=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            SVClassifier = SVClassfication(ngram_range=ngram_range, max_features=max_features, classifier=SVC(C=bestparams['C'], kernel='linear', probability=True))
            SVClassifier.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = SVClassifier.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            pred_probs = SVClassifier.predict_proba(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():
                score = SVClassifier.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return SVClassifier, preds, pred_probs, score
            else:
                return SVClassifier, preds, pred_probs
        if kneighborsclassifier:
            KNClassifier = KNeighborsC(ngram_range=ngram_range, max_features=max_features)
            bestscore, bestparams = gscv(KNClassifier, {'p': [2, 3, 4, 5]}, 5, X_train, y_train,
                classifier=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            KNClassifier = KNeighborsC(ngram_range=ngram_range, max_features=max_features, classifier=KNeighborsClassifier(n_neighbors=2, p=bestparams['p']))
            KNClassifier.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = KNClassifier.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            pred_probs = KNClassifier.predict_proba(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():
                score = KNClassifier.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return KNClassifier, preds, pred_probs, score
            else:
                return KNClassifier, preds, pred_probs
        if probability:
            if bernoullinb:
                BNBClassifier = BernoulliNaiveBayes(ngram_range=ngram_range, max_features=max_features)
                BNBClassifier.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                preds = BNBClassifier.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                pred_probs = BNBClassifier.predict_proba(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                if y_test.any():   
                    score = BNBClassifier.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                    return BNBClassifier, preds, pred_probs, score
                else:
                    return BNBClassifier, preds, pred_probs
            if multinomialnb:
                MNBClassifier = MultinomialNaiveBayes(ngram_range=ngram_range, max_features=max_features)
                MNBClassifier.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                preds = MNBClassifier.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                pred_probs = MNBClassifier.predict_proba(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                if y_test.any():   
                    score = MNBClassifier.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                    return MNBClassifier, preds, pred_probs, score
                else:
                    return MNBClassifier, preds, pred_probs
            if gaussiannb:
                GNBClassifier = GaussianNaiveBayes(ngram_range=ngram_range, max_features=max_features)
                GNBClassifier.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                preds = GNBClassifier.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                pred_probs = GNBClassifier.predict_proba(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                if y_test.any():                 
                    score = GNBClassifier.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                    return GNBClassifier, preds, pred_probs, score
                else:
                    return GNBClassifier, preds, pred_probs
            if basehmm:
                BHMMClassifier = BaseHiddenMarkovModels(ngram_range=ngram_range, max_features=max_features, classifier=_BaseHMM(n_components=5))
                BHMMClassifier.fit(X_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                preds = BHMMClassifier.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                pred_probs = BHMMClassifier.predict_proba(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                score = BHMMClassifier.score(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return BHMMClassifier, preds, pred_probs, score
            if multinomialhmm:
                MHMMClassifier = MultinomialHiddenMarkovModels(ngram_range=ngram_range, max_features=max_features, classifier=MultinomialHMM(n_components=5))
                MHMMClassifier.fit(X_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                preds = MHMMClassifier.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                pred_probs = MHMMClassifier.predict_proba(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                score = MHMMClassifier.score(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return MHMMClassifier, preds, pred_probs, score
            if gaussianhmm:
                GHMMClassifier = GaussianHiddenMarkovModels(ngram_range=ngram_range, max_features=max_features, classifier=GaussianHMM(n_components=5))
                GHMMClassifier.fit(X_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                preds = GHMMClassifier.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                pred_probs = GHMMClassifier.predict_proba(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                score = GHMMClassifier.score(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return GHMMClassifier, preds, pred_probs, score
            if gmmhmm:
                GMMHMMClassifier = GaussianMixtureEmissionsHiddenMarkovModels(ngram_range=ngram_range, max_features=max_features, classifier=GMMHMM(n_components=5))
                GMMHMMClassifier.fit(X_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                preds = GMMHMMClassifier.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                pred_probs = GMMHMMClassifier.predict_proba(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                score = GMMHMMClassifier.score(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return GMMHMMClassifier, preds, pred_probs, score
        if baggingclassifier:
            LRClassifier = LogisticR(ngram_range=ngram_range, max_features=max_features)
            BGClassifier = BaggingC(ngram_range=ngram_range, max_features=max_features, classifier=BaggingClassifier(base_estimator=LRClassifier.classifier))
            bestscore, bestparams = gscv(BGClassifier, {'n_estimators': [10, 50, 120, 200, 300, 500]}, 5, X_train, y_train,
                classifier=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            BGClassifier = BaggingC(ngram_range=ngram_range, max_features=max_features, classifier=BaggingClassifier(base_estimator=LRClassifier.classifier, n_estimators=bestparams['n_estimators']))
            BGClassifier.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = BGClassifier.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            pred_probs = BGClassifier.predict_proba(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():              
                score = BGClassifier.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return BGClassifier, preds, pred_probs, score
            else:
                return BGClassifier, preds, pred_probs
        if randomforestclassifier:
            RFClassifier = RandomForestC(ngram_range=ngram_range, max_features=max_features)
            bestscore, bestparams = gscv(RFClassifier, {'n_estimators': [10, 50, 120, 200],
                                                        'max_depth': [5, 8, 15, 25],
                                                        'min_samples_split': [2, 5, 10, 15],
                                                        'min_samples_leaf': [1, 2, 5, 10]}, 5, X_train, y_train,
                                                        classifier=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            RFClassifier = RandomForestC(ngram_range=ngram_range, max_features=max_features, classifier=RandomForestClassifier(n_estimators=bestparams['n_estimators'],
                                                     max_depth=bestparams['max_depth'], 
                                                     min_samples_split=bestparams['min_samples_split'],
                                                     min_samples_leaf=bestparams['min_samples_leaf']))
            RFClassifier.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = RFClassifier.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            pred_probs = RFClassifier.predict_proba(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():
                score = RFClassifier.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return RFClassifier, preds, pred_probs, score
            else:
                return RFClassifier, preds, pred_probs
        if adaboostclassifier:
            LRClassifier = LogisticR(ngram_range=ngram_range, max_features=max_features)
            ABClassifier = AdaBoostC(ngram_range=ngram_range, max_features=max_features, classifier=AdaBoostClassifier(base_estimator=LRClassifier.classifier))
            bestscore, bestparams = gscv(ABClassifier, {'n_estimators': [10, 50, 120, 200, 300, 500],
                                                        'learning_rate': [0.001, 0.01, 0.05, 0.1, 1.0, 10.0]}, 5, X_train, y_train,
                                                        classifier=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            ABClassifier = AdaBoostC(ngram_range=ngram_range, max_features=max_features, classifier=AdaBoostClassifier(base_estimator=LRClassifier.classifier,
                                                        n_estimators=bestparams['n_estimators'],
                                                        learning_rate=bestparams['learning_rate']))
            ABClassifier.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = ABClassifier.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            pred_probs = ABClassifier.predict_proba(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():            
                score = ABClassifier.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return ABClassifier, preds, pred_probs, score
            else:
                return ABClassifier, preds, pred_probs
        if gradientboostingclassifier:
            GBClassifier = GradientBoostingC(ngram_range=ngram_range, max_features=max_features)
            bestscore, bestparams = gscv(GBClassifier, {'learning_rate': [0.05, 0.1, 0.15, 0.2],
                                                        'n_estimators': [40, 50, 60, 70],
                                                        'max_depth': [3, 5, 7, 9],
                                                        'min_samples_split': [2, 5, 10, 15],
                                                        'min_samples_leaf': [1, 2, 5, 10],
                                                        'subsample': [0.6, 0.7, 0.8, 0.9]}, 5, X_train, y_train,
                                                        classifier=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            GBClassifier = GradientBoostingC(ngram_range=ngram_range, max_features=max_features, classifier=GradientBoostingClassifier(n_estimators=bestparams['n_estimators'],
                                                                                  max_depth=bestparams['max_depth'], 
                                                                                  min_samples_split=bestparams['min_samples_split'], 
                                                                                  min_samples_leaf=bestparams['min_samples_leaf'],
                                                                                  learning_rate=bestparams['learning_rate'], 
                                                                                  subsample=bestparams['subsample']))
            GBClassifier.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = GBClassifier.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            pred_probs = GBClassifier.predict_proba(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():               
                score = GBClassifier.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return GBClassifier, preds, pred_probs, score
            else:
                return GBClassifier, preds, pred_probs
        if xgbclassifier:
            XGBClassification = XGBC(ngram_range=ngram_range, max_features=max_features)
            bestscore, bestparams = gscv(XGBClassification, {'learning_rate': [0.05, 0.1, 0.15, 0.2],
                                                             'n_estimators': [40, 50, 60, 70],
                                                             'max_depth': [3, 5, 7, 9],
                                                             'min_child_weight': [1, 3, 5, 7],
                                                             'subsample': [0.6, 0.7, 0.8, 0.9],
                                                             'gamma': [0.05, 0.1, 0.3, 0.5],
                                                             'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                                                            # 'reg_lambda': [0.01, 0.05, 0.1, 1.0],
                                                            # 'reg_alpha': [0, 0.1, 0.5, 1.0]
                                                              }, 5, X_train, y_train, classifier=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            XGBClassification = XGBC(ngram_range=ngram_range, max_features=max_features, classifier=XGBClassifier(n_estimators=bestparams['n_estimators'], 
                                                              max_depth=bestparams['max_depth'], 
                                                              gamma=bestparams['gamma'], 
                                                              min_child_weight=bestparams['min_child_weight'], 
                                                              subsample=bestparams['subsample'],
                                                              colsample_bytree=bestparams['colsample_bytree'], 
                                                             # reg_lambda=bestparams['reg_lambda'], 
                                                             # reg_alpha=bestparams['reg_alpha'],
                                                              learning_rate=bestparams['learning_rate']))
            XGBClassification.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = XGBClassification.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            pred_probs = XGBClassification.predict_proba(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():            
                score = XGBClassification.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return XGBClassification, preds, pred_probs, score
            else:
                return XGBClassification, preds, pred_probs
        if lgbmclassifier:
            LGBMClassification = LGBMC(ngram_range=ngram_range, max_features=max_features)
            bestscore, bestparams = gscv(LGBMClassification, {'learning_rate': [0.05, 0.1, 0.15, 0.2],
                                                                   'n_estimators': [40, 50, 60, 70],
                                                                   'max_depth': [3, 5, 7, 9],
                                                                   'num_leaves': [10, 20, 30, 50],
                                                                   'min_child_samples': [1, 3, 5, 7],
                                                                   'min_child_weight': [0.001, 0.005, 0.01, 0.05],
                                                                   'subsample': [0.6, 0.7, 0.8, 0.9],
                                                                   'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                                                                  # 'reg_lambda': [0.01, 0.05, 0.1, 1.0],
                                                                  # 'reg_alpha': [0, 0.1, 0.5, 1.0]
                                                                    }, 5, X_train, y_train, classifier=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            LGBMClassification = LGBMC(ngram_range=ngram_range, max_features=max_features, classifier=LGBMClassifier(n_estimators=bestparams['n_estimators'], 
                                                                max_depth=bestparams['max_depth'], 
                                                                colsample_bytree=bestparams['colsample_bytree'], 
                                                                min_child_weight=bestparams['min_child_weight'], 
                                                                min_child_samples=bestparams['min_child_samples'],
                                                                subsample=bestparams['subsample'], 
                                                                num_leaves=bestparams['num_leaves'], 
                                                               # reg_lambda=bestparams['reg_lambda'], 
                                                               # reg_alpha=bestparams['reg_alpha'],
                                                                learning_rate=bestparams['learning_rate']))
            LGBMClassification.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = LGBMClassification.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            pred_probs = LGBMClassification.predict_proba(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():
                score = LGBMClassification.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return LGBMClassification, preds, pred_probs, score
            else:
                return LGBMClassification, preds, pred_probs
        if mlpclassifier:
            MLPClassification = MLPC(ngram_range=ngram_range, max_features=max_features)
            bestscore, bestparams = gscv(MLPClassification, {'hidden_layer_sizes': [(100,), (300,), (500,), (1000,)],
                                                             'alpha': [0.0001, 0.0002, 0.0005, 0.001],
                                                             'learning_rate_init': [0.001, 0.002, 0.005, 0.01],
                                                             'max_iter': [100, 200, 300, 500]}, 5, X_train, y_train,
                                                             classifier=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            MLPClassification = MLPC(ngram_range=ngram_range, max_features=max_features, classifier=MLPClassifier(hidden_layer_sizes=bestparams['hidden_layer_sizes'], 
                                                              alpha=bestparams['alpha'],
                                                              learning_rate_init=bestparams['learning_rate_init'],
                                                              max_iter=bestparams['max_iter']))
            MLPClassification.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = MLPClassification.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            pred_probs = MLPClassification.predict_proba(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():            
                score = MLPClassification.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return MLPClassification, preds, pred_probs, score
            else:
                return MLPClassification, preds, pred_probs
        if tpotclassifier:
            TPOTClassification = TPOTC(ngram_range=ngram_range, max_features=max_features, classifier=TPOTClassifier(generations=10, population_size=50, verbosity=2, n_jobs=-1))
            TPOTClassification.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = TPOTClassification.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            pred_probs = TPOTClassification.predict_proba(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():
                score = TPOTClassification.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return TPOTClassification, preds, pred_probs, score
            else:
                return TPOTClassification, preds, pred_probs
    if regression:
        if linearregression:
            LRegressor = LinearR(ngram_range=ngram_range, max_features=max_features)
            LRegressor.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = LRegressor.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():            
                score = LRegressor.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return LRegressor, preds, score
            else:
                return LRegressor, preds
        if decisiontreeregressor:
            DTRegressor = DecisionTreeR(ngram_range=ngram_range, max_features=max_features)
            bestscore, bestparams = gscv(DTRegressor, {'max_depth': [5, 8, 15, 25],
                                                      'min_samples_split': [2, 5, 10, 15],
                                                      'min_samples_leaf': [1, 2, 5, 10]}, 5, X_train, y_train,
                                                      regressor=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            DTRegressor = DecisionTreeR(ngram_range=ngram_range, max_features=max_features, regressor=DecisionTreeRegressor(max_depth=bestparams['max_depth'], min_samples_split=bestparams['min_samples_split'], 
                                                           min_samples_leaf=bestparams['min_samples_leaf']))
            DTRegressor.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = DTRegressor.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():               
                score = DTRegressor.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return DTRegressor, preds, score
            else:
                return DTRegressor, preds
        if svr:
            SVRegressor = SVRegression(ngram_range=ngram_range, max_features=max_features)
            bestscore, bestparams = gscv(SVRegressor, {'C': [0.01, 0.1, 1.0, 10.0, 50.0, 100.0],
                'epsilon': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],'max_iter':[100, 500, 1000, 2000, 3000, 5000]}, 5, X_train, y_train,
                regressor=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            SVRegressor = SVRegression(ngram_range=ngram_range, max_features=max_features, regressor=SVR(C=bestparams['C'], kernel='linear'))
            SVRegressor.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = SVRegressor.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():             
                score = SVRegressor.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return SVRegressor, preds, score
            else:
                return SVRegressor, preds
        if kneighborsregressor:
            KNRegressor = KNeighborsR(ngram_range=ngram_range, max_features=max_features)
            bestscore, bestparams = gscv(KNRegressor, {'leaf_size': [10, 20, 30, 50],
                                                        'p': [2, 3, 4, 5]}, 5, X_train, y_train,
                                                        regressor=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            KNRegressor = KNeighborsR(ngram_range=ngram_range, max_features=max_features, regressor=KNeighborsRegressor(n_neighbors=2, leaf_size=bestparams['leaf_size'], p=bestparams['p']))
            KNRegressor.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = KNRegressor.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():              
                score = KNRegressor.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return KNRegressor, preds, score
            else:
                return KNRegressor, preds
        if baggingregressor:
            LRegressor = LinearR(ngram_range=ngram_range, max_features=max_features)
            BGRegressor = BaggingR(ngram_range=ngram_range, max_features=max_features, regressor=BaggingRegressor(base_estimator=LRegressor.regressor))
            bestscore, bestparams = gscv(BGRegressor, {'n_estimators': [10, 50, 120, 200, 300, 500]}, 5, X_train, y_train,
                 regressor=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            BGRegressor = BaggingR(ngram_range=ngram_range, max_features=max_features, regressor=BaggingRegressor(base_estimator=LRegressor.regressor, n_estimators=bestparams['n_estimators']))
            BGRegressor.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = BGRegressor.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():              
                score = BGRegressor.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return BGRegressor, preds, score
            else:
                return BGRegressor, preds
        if randomforestregressor:
            RFRegressor = RandomForestR(ngram_range=ngram_range, max_features=max_features)
            bestscore, bestparams = gscv(RFRegressor, {'n_estimators': [10, 50, 120, 200],
                                                                  'max_depth': [5, 8, 15, 25],
                                                                  'min_samples_split': [2, 5, 10, 15],
                                                                  'min_samples_leaf': [1, 2, 5, 10]}, 5, X_train, y_train,
                                                                   regressor=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            RFRegressor = RandomForestR(ngram_range=ngram_range, max_features=max_features, regressor=RandomForestRegressor(n_estimators=bestparams['n_estimators'], max_depth=bestparams['max_depth'], 
                                                   min_samples_split=bestparams['min_samples_split'], min_samples_leaf=bestparams['min_samples_leaf']))
            RFRegressor.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = RFRegressor.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():             
                score = RFRegressor.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return RFRegressor, preds, score
            else:
                return RFRegressor, preds
        if adaboostregressor:
            LRegressor = LinearR(ngram_range=ngram_range, max_features=max_features)
            ABRegressor = AdaBoostR(ngram_range=ngram_range, max_features=max_features, regressor=AdaBoostRegressor(base_estimator=LRegressor.regressor))
            bestscore, bestparams = gscv(ABRegressor, {'n_estimators': [10, 50, 120, 200, 300, 500],
                                                        'learning_rate': [0.001, 0.01, 0.05, 0.1, 1.0, 10.0]}, 5, X_train, y_train,
                                                        regressor=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            ABRegressor = AdaBoostR(ngram_range=ngram_range, max_features=max_features, regressor=AdaBoostRegressor(base_estimator=LRegressor.regressor, n_estimators=bestparams['n_estimators'], 
                                                          learning_rate=bestparams['learning_rate']))
            ABRegressor.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = ABRegressor.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():            
                score = ABRegressor.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return ABRegressor, preds, score
            else:
                return ABRegressor, preds
        if gradientboostingregressor:
            GBRegressor = GradientBoostingR(ngram_range=ngram_range, max_features=max_features)
            bestscore, bestparams = gscv(GBRegressor, {'learning_rate': [0.05, 0.1, 0.15, 0.2],
                                                       'n_estimators': [40, 50, 60, 70],
                                                       'max_depth': [3, 5, 7, 9],
                                                       'min_samples_split': [2, 5, 10, 15],
                                                       'min_samples_leaf': [1, 2, 5, 10],
                                                       'subsample': [0.6, 0.7, 0.8, 0.9],
                                                       'alpha': [0, 0.1, 0.5, 1.0]}, 5, X_train, y_train,
                                                       regressor=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            GBRegressor = GradientBoostingR(ngram_range=ngram_range, max_features=max_features, regressor=GradientBoostingRegressor(n_estimators=bestparams['n_estimators'],
                                                                                max_depth=bestparams['max_depth'], 
                                                                                min_samples_split=bestparams['min_samples_split'], 
                                                                                min_samples_leaf=bestparams['min_samples_leaf'],
                                                                                learning_rate=bestparams['learning_rate'], 
                                                                                subsample=bestparams['subsample'],
                                                                                alpha=bestparams['alpha']))
            GBRegressor.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = GBRegressor.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():      
                score = GBRegressor.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return GBRegressor, preds, score
            else:
                return GBRegressor, preds
        if xgbregressor:
            XGBRegression = XGBR(ngram_range=ngram_range, max_features=max_features)
            bestscore, bestparams = gscv(XGBRegression, {'learning_rate': [0.05, 0.1, 0.15, 0.2],
                                                                   'n_estimators': [40, 50, 60, 70],
                                                                   'max_depth': [3, 5, 7, 9],
                                                                   'min_child_weight': [1, 3, 5, 7],
                                                                   'subsample': [0.6, 0.7, 0.8, 0.9],
                                                                   'gamma': [0.05, 0.1, 0.3, 0.5],
                                                                   'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                                                                  # 'reg_lambda': [0.01, 0.05, 0.1, 1.0],
                                                                  # 'reg_alpha': [0, 0.1, 0.5, 1.0]
                                                                  }, 5, X_train, y_train, regressor=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            XGBRegression = XGBR(ngram_range=ngram_range, max_features=max_features, regressor=XGBRegressor(n_estimators=bestparams['n_estimators'], 
                                                          max_depth=bestparams['max_depth'], 
                                                          gamma=bestparams['gamma'], 
                                                          min_child_weight=bestparams['min_child_weight'], 
                                                          subsample=bestparams['subsample'],
                                                          colsample_bytree=bestparams['colsample_bytree'], 
                                                         # reg_lambda=bestparams['reg_lambda'], 
                                                         # reg_alpha=bestparams['reg_alpha'],
                                                          learning_rate=bestparams['learning_rate']))
            XGBRegression.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer,
                early_stopping_rounds=early_stopping_rounds, eval_metric=eval_metric, eval_set=eval_set)
            preds = XGBRegression.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():              
                score = XGBRegression.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return XGBRegression, preds, score
            else:
                return XGBRegression, preds
        if lgbmregressor:
            LGBMRegression = LGBMR(ngram_range=ngram_range, max_features=max_features)
            bestscore, bestparams = gscv(LGBMRegression, {'learning_rate': [0.05, 0.1, 0.15, 0.2],
                                                          'n_estimators': [40, 50, 60, 70],
                                                          'max_depth': [3, 5, 7, 9],
                                                          'num_leaves': [10, 20, 30, 50],
                                                          'min_child_samples': [1, 3, 5, 7],
                                                          'min_child_weight': [0.001, 0.005, 0.01, 0.05],
                                                          'subsample': [0.6, 0.7, 0.8, 0.9],
                                                          'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                                                         # 'reg_lambda': [0.01, 0.05, 0.1, 1.0],
                                                         # 'reg_alpha': [0, 0.1, 0.5, 1.0]
                                                           }, 5, X_train, y_train, regressor=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            LGBMRegression = LGBMR(ngram_range=ngram_range, max_features=max_features, regressor=LGBMRegressor(n_estimators=bestparams['n_estimators'], 
                                                            max_depth=bestparams['max_depth'], 
                                                            colsample_bytree=bestparams['colsample_bytree'], 
                                                            min_child_weight=bestparams['min_child_weight'], 
                                                            min_child_samples=bestparams['min_child_samples'],
                                                            subsample=bestparams['subsample'], 
                                                            num_leaves=bestparams['num_leaves'], 
                                                           # reg_lambda=bestparams['reg_lambda'], 
                                                           # reg_alpha=bestparams['reg_alpha'],
                                                            learning_rate=bestparams['learning_rate']))
            LGBMRegression.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer,
                early_stopping_rounds=early_stopping_rounds, eval_metric=eval_metric, eval_set=eval_set)
            preds = LGBMRegression.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():                
                score = LGBMRegression.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return LGBMRegression, preds, score
            else:
                return LGBMRegression, preds
        if mlpregressor:
            MLPRegression = MLPR(ngram_range=ngram_range, max_features=max_features)
            bestscore, bestparams = gscv(MLPRegression, {'hidden_layer_sizes': [(100,), (300,), (500,)],
                                                         'alpha': [0.0001, 0.0002, 0.0005],
                                                         'learning_rate_init': [0.001, 0.002, 0.005],
                                                         'max_iter': [100, 200, 300]}, 5, X_train, y_train,
                                                          regressor=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            MLPRegression = MLPR(ngram_range=ngram_range, max_features=max_features, regressor=MLPRegressor(hidden_layer_sizes=bestparams['hidden_layer_sizes'], 
                                                          alpha=bestparams['alpha'],
                                                          learning_rate_init=bestparams['learning_rate_init'],
                                                          max_iter=bestparams['max_iter']))
            MLPRegression.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = MLPRegression.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():              
                score = MLPRegression.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return MLPRegression, preds, score
            else:
                return MLPRegression, preds
        if tpotregressor:
            TPOTRegression = TPOTR(ngram_range=ngram_range, max_features=max_features, regressor=TPOTRegressor(generations=100, population_size=100, verbosity=2, n_jobs=-1))
            TPOTRegression.fit(X_train, y_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = TPOTRegression.predict(X_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            if y_test.any():               
                score = TPOTRegression.score(X_test, y_test, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
                return TPOTRegression, preds, score
            else:
                return TPOTRegression, preds
    if clustering:
        if kmeans:
            KMCluster = KmeansClustering(ngram_range=ngram_range, max_features=max_features)
            bestscore, bestparams = gscv(KMCluster, {'random_state': [0, 50, 100, 200, 500, 800, 1000],
                                                    'max_iter': [300, 500, 1000, 3000]}, 5, X_train, y_train,
                                                     cluster=True, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            KMCluster = KmeansClustering(ngram_range=ngram_range, max_features=max_features, cluster=KMeans(n_clusters=10, init='random',
                                                     random_state=bestparams['random_state'], 
                                                     max_iter=bestparams['max_iter']))
            KMCluster.fit(X_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = KMCluster.predict(X_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            score = KMCluster.score(X_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            return KMCluster, preds, score
        if spectralclustering:
            SCluster = SClustering(ngram_range=ngram_range, max_features=max_features, cluster=SpectralClustering(n_clusters=10, affinity='rbf', n_neighbors=10, assign_labels='kmeans'))
            SCluster.fit(X_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            preds = SCluster.fit_predict(X_train, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
            return SCluster, preds
    if decomposition:
        if pca:
            PCAdecomposition = PrincipalComponentAnalysis(decomposition=PCA(n_components=2))
            results = PCAdecomposition.fit_transform(X_train)
            return results
        if nmf:
            NFMdecomposition = NonNegativeMatrixFactorization(decomposition=NMF(n_components=2, max_iter=200, 
                                         random_state=None, shuffle=False))
            results = PCAdecomposition.fit_transform(X_train)
            return results
        if isomap:
            IsoMapdecomposition = IsometricMapping(decomposition=IsoMap(n_neighbors=5, n_components=2))
            results = IsoMapdecomposition.fit_transform(X_train)
            return results
        if lda:
            LDAdecomposition = LDAllocation(decomposition=LatentDirichletAllocation(n_components=10, max_iter=20,
                                        total_samples=10000.0, random_state=None, n_topics=5))
            results = LDAdecomposition.fit_transform(X_train)
            return results           
    if association:
        if aprior:
            freqslist, freqssupp = apriori(transactions, threshold)
            association_rules = generateRules(freqslist, freqssupp, threshold)
            return freqslist, freqssupp, association_rules
        if fp_growth:
            freqslist = []
            for freqs in find_frequent_itemsets(transactions, threshold, include_support=True):
                freqslist.append(freqs)
            return freqslist


# In[21]:


# 逻辑回归
class LogisticR(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, classifier=LogisticRegression(C=1.0, multi_class='ovr', class_weight=None, n_jobs=-1)):
        self.classifier = classifier
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.classifier.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def predict_proba(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict_proba(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# 决策树分类
class DecisionTreeC(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, classifier=DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                                                     max_features=None, class_weight=None)):
        self.classifier = classifier
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.classifier.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def predict_proba(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict_proba(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# 支持向量分类
class SVClassfication(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, classifier=SVC(C=1.0, kernel='rbf', probability=True, decision_function_shape='ovo', class_weight=None)):
        self.classifier = classifier
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.classifier.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def predict_proba(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict_proba(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# 最近邻分类
class KNeighborsC(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, classifier=KNeighborsClassifier(n_neighbors=2, p=2, n_jobs=-1)):
        self.classifier = classifier
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.classifier.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def predict_proba(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict_proba(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)

# 伯努利分布朴素贝叶斯分类
class BernoulliNaiveBayes(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, classifier=BernoulliNB()):
        self.classifier = classifier
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.classifier.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer).toarray(), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer).toarray())
    
    def predict_proba(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict_proba(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer).toarray())
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer).toarray(), y)

# 多项分布朴素贝叶斯分类
class MultinomialNaiveBayes(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, classifier=MultinomialNB()):
        self.classifier = classifier
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.classifier.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer).toarray(), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer).toarray())
    
    def predict_proba(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict_proba(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer).toarray())
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer).toarray(), y)
    
# 多项分布朴素贝叶斯分类
class GaussianNaiveBayes(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, classifier=GaussianNB()):
        self.classifier = classifier
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.classifier.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer).toarray(), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer).toarray())
    
    def predict_proba(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict_proba(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer).toarray())
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer).toarray(), y)
    
# 基础版隐马尔可夫
class BaseHiddenMarkovModels(object):

    def __init__(self,  ngram_range=(1, 1), max_features=None, classifier=_BaseHMM(n_components=5)):
        self.classifier = classifier
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            self.labelencoder.fit(X)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.classifier.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def predict_proba(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict_proba(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def startprob_(self):
        return self.classifier.startprob_
    
    def transmat_(self):
        return self.classifier.transmat_
    
# 多分类隐马尔可夫
class MultinomialHiddenMarkovModels(object):

    def __init__(self,  ngram_range=(1, 1), max_features=None, classifier=MultinomialHMM(n_components=5)):
        self.classifier = classifier
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            self.labelencoder.fit(X)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.classifier.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def predict_proba(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict_proba(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def startprob_(self):
        return self.classifier.startprob_
    
    def transmat_(self):
        return self.classifier.transmat_
    
    def emissionprob_(self):
        return self.classifier.emissionprob_
    
# 高斯隐马尔可夫
class GaussianHiddenMarkovModels(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, classifier=GaussianHMM(n_components=5)):
        self.classifier = classifier
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            self.labelencoder.fit(X)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.classifier.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def predict_proba(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict_proba(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def startprob_(self):
        return self.classifier.startprob_
    
    def transmat_(self):
        return self.classifier.transmat_
    
    def means_(self):
        return self.classifier.means_
    
    def covars_(self):
        return self.classifier.covars_
    
# 混合高斯隐马尔可夫
class GaussianMixtureEmissionsHiddenMarkovModels(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, classifier=GMMHMM(n_components=5)):
        self.classifier = classifier
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            self.labelencoder.fit(X)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.classifier.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def predict_proba(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict_proba(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def startprob_(self):
        return self.classifier.startprob_
    
    def transmat_(self):
        return self.classifier.transmat_
    
    def weights_(self):
        return self.classifier.weights_
    
    def means_(self):
        return self.classifier.means_
    
    def covars_(self):
        return self.classifier.covars_
    
# Bagging分类
class BaggingC(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, classifier=BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, 
                                                    max_features=1.0, oob_score=True, n_jobs=-1)):
        self.classifier = classifier
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if yNone:
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if yNone:
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.classifier.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def predict_proba(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict_proba(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# 随机森林分类
class RandomForestC(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, classifier=RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, 
                                                         min_samples_leaf=1, max_features='auto', oob_score=True, n_jobs=-1, class_weight=None)):
        self.classifier = classifier
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.classifier.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def predict_proba(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict_proba(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# AdaBoost分类
class AdaBoostC(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, classifier=AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0)):
        self.classifier = classifier
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.classifier.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def predict_proba(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict_proba(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# GBDT分类
class GradientBoostingC(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, classifier=GradientBoostingClassifier(criterion='friedman_mse', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, 
                                                             min_samples_leaf=1, max_depth=3, max_features=None)):
        self.classifier = classifier
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.classifier.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def predict_proba(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict_proba(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# XGBoost分类
class XGBC(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, classifier=XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, objective='binary:logistic', booster='gbtree',
                                               n_jobs=1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
                                               reg_alpha=0, reg_lambda=1, scale_pos_weight=1)):
        self.classifier = classifier
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False, early_stopping_rounds=None, eval_metric=None, eval_set=None):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.classifier.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y,
            early_stopping_rounds=early_stopping_rounds, eval_metric=eval_metric, eval_set=eval_set)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def predict_proba(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict_proba(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# LightGBM分类
class LGBMC(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, classifier=LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, 
                                               class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, 
                                               colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, n_jobs=1)):
        self.classifier = classifier
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False, early_stopping_rounds=None, eval_metric=None, eval_set=None):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.classifier.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y,
            early_stopping_rounds=early_stopping_rounds, eval_metric=eval_metric, eval_set=eval_set)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def predict_proba(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict_proba(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        return self.classifier.score(x, y)
    
# 多层神经网络分类
class MLPC(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, classifier=MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, 
                                              learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, 
                                              shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, 
                                              momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, 
                                              beta_1=0.9, beta_2=0.999, epsilon=1e-08)):
        self.classifier = classifier
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.classifier.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def predict_proba(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.predict_proba(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.classifier.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# TPOT分类
class TPOTC(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, classifier=TPOTClassifier(generations=10, population_size=50, verbosity=2, n_jobs=-1)):
        self.classifier = classifier
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)

    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        if isinstance(X, pd.DataFrame):
            self.classifier.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        else:
            self.classifier.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer).toarray(), np.array(y))
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if isinstance(x, pd.DataFrame):
            return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
        else:
            return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer).toarray())
    
    def predict_proba(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if isinstance(x, pd.DataFrame):
            return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
        else:
            return self.classifier.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer).toarray())
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if isinstance(x, pd.DataFrame):
            return self.classifier.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        else:
            return self.classifier.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer).toarray(), np.array(y))


# In[22]:


# 线性回归
class LinearR(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, regressor=LinearRegression(n_jobs=-1)):
        self.regressor = regressor
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.regressor.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# 树回归
class DecisionTreeR(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, regressor=DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, 
                                                       min_samples_leaf=1, max_features=None, presort=True)):
        self.regressor = regressor
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.regressor.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# 支持向量回归
class SVRegression(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, regressor=SVR(C=1.0, kernel='rbf', epsilon=0.1, max_iter=-1)):
        self.regressor = regressor
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.regressor.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# 最近邻回归
class KNeighborsR(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, regressor=KNeighborsRegressor(n_neighbors=2, leaf_size=30, p=2, n_jobs=-1)):
        self.regressor = regressor
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.regressor.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# Bagging回归
class BaggingR(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, regressor=BaggingRegressor(base_estimator=None, n_estimators=10, max_samples=1.0, 
                                                    max_features=1.0, oob_score=True, n_jobs=-1)):
        self.regressor = regressor
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.regressor.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# 随机森林回归
class RandomForestR(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, regressor=RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, 
                                                         min_samples_leaf=1, max_features='auto', oob_score=True, n_jobs=-1, random_state=None)):
        self.regressor = regressor
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.regressor.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# AdaBoost回归
class AdaBoostR(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, regressor=AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0)):
        self.regressor = regressor
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.regressor.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# GBDT回归
class GradientBoostingR(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, regressor=GradientBoostingRegressor(loss='ls', criterion='friedman_mse', random_state=None, 
                                                           learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, 
                                                             min_samples_leaf=1, max_depth=3, alpha=0.9, max_features=None)):
        self.regressor = regressor
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.regressor.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# XGBoost回归
class XGBR(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, regressor=XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, objective='reg:linear', booster='gbtree',
                                               n_jobs=1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
                                               reg_alpha=0, reg_lambda=1, scale_pos_weight=1)):
        self.regressor = regressor
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False, early_stopping_rounds=None, eval_metric=None, eval_set=None):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.regressor.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y,
            early_stopping_rounds=early_stopping_rounds, eval_metric=eval_metric, eval_set=eval_set)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# LightGBM回归
class LGBMR(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, regressor=LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, 
                                               class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, 
                                               colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, n_jobs=1)):
        self.regressor = regressor
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False, early_stopping_rounds=None, eval_metric=None, eval_set=None):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.regressor.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y,
            early_stopping_rounds=early_stopping_rounds, eval_metric=eval_metric, eval_set=eval_set)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# 多层神经网络回归
class MLPR(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, regressor=MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, 
                                              learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, 
                                              shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, 
                                              momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, 
                                              beta_1=0.9, beta_2=0.999, epsilon=1e-08)):
        self.regressor = regressor
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.regressor.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.regressor.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# TPOT回归
class TPOTR(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, regressor=TPOTRegressor(generations=100, population_size=100, verbosity=2, n_jobs=-1)):
        self.regressor = regressor
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)

    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        if isinstance(X, pd.DataFrame):
            self.regressor.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        else:
            self.regressor.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer).toarray(), np.array(y))
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if isinstance(x, pd.DataFrame):
            return self.regressor.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
        else:
            return self.regressor.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer).toarray())
    
    def predict_proba(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if isinstance(x, pd.DataFrame):
            return self.regressor.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
        else:
            return self.regressor.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer).toarray())
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if isinstance(x, pd.DataFrame):
            return self.regressor.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        else:
            return self.regressor.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer).toarray(), np.array(y))


# In[23]:


# Kmeans聚类
class KmeansClustering(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, cluster=KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=300, 
                                      random_state=None, n_jobs=-1)):
        self.cluster = cluster
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.cluster.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.cluster.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.cluster.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
    
# 层次聚类
class SClustering(object):

    def __init__(self, ngram_range=(1, 1), max_features=None, cluster=SpectralClustering(n_clusters=10, random_state=None, n_init=10, gamma=1.0, n_neighbors=10, 
                                                  eigen_tol=0.0, degree=3, coef0=1, kernel_params=None, n_jobs=-1)):
        self.cluster = cluster
        self.stdscaler = StandardScaler()
        self.onehotencoder = OneHotEncoder()
        self.labelencoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)
        
    def features(self, X, y=None, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            return self.stdscaler.transform(X)
        elif onehotencoder:
            return self.onehotencoder.transform(X)
        elif labelencoder:
            if y.any():
                return self.labelencoder.transform(y)
        elif vectorizer:
            return self.vectorizer.transform(X)
        else:
            return X
    
    def fit(self, X, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        if stdscaler:
            self.stdscaler.fit(X)
        elif onehotencoder:
            self.onehotencoder.fit(X)
        elif labelencoder:
            if y.any():
                self.labelencoder.fit(y)
        elif vectorizer:
            self.vectorizer.fit(X)
        self.cluster.fit(self.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)
        
    def predict(self, x, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.cluster.predict(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer))
    
    def score(self, x, y, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
        return self.cluster.score(self.features(x, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer), y)


# In[24]:


# PCA线性降维
class PrincipalComponentAnalysis(object):

    def __init__(self, decomposition=PCA(n_components=None, random_state=None)):
        self.decomposition = decomposition
        
    def fit(self, X):
        self.decomposition.fit(X)
        
    def transform(self, X):
        self.decomposition.transform(X)
    
    def fit_transform(self, X):
        self.decomposition.fit_transform(X)
        
    def inverse_transform(self, X):
        self.decomposition.inverse_transform(X)
        
# NMF非负矩阵分解
class NonNegativeMatrixFactorization(object):

    def __init__(self, decomposition=NMF(n_components=None, beta_loss='frobenius', tol=0.0001, max_iter=200, 
                                         random_state=None, alpha=0.0, l1_ratio=0.0, shuffle=False)):
        self.decomposition = decomposition
        
    def fit(self, X):
        self.decomposition.fit(X)
        
    def transform(self, X):
        self.decomposition.transform(X)
    
    def fit_transform(self, X):
        self.decomposition.fit_transform(X)
        
    def inverse_transform(self, X):
        self.decomposition.inverse_transform(X)
        
# IsoMap非线性降维
class IsometricMapping(object):

    def __init__(self, decomposition=Isomap(n_neighbors=5, n_components=2, max_iter=None, n_jobs=-1)):
        self.decomposition = decomposition
        
    def fit(self, X):
        self.decomposition.fit(X)
        
    def transform(self, X):
        self.decomposition.transform(X)
    
    def fit_transform(self, X):
        self.decomposition.fit_transform(X)
        
    def inverse_transform(self, X):
        self.decomposition.inverse_transform(X)
        
# LDA主题模型
class LDAllocation(object):

    def __init__(self, decomposition=LatentDirichletAllocation(n_components=10, max_iter=10, total_samples=1000000.0,
                                                              random_state=None, n_topics=None, n_jobs=-1)):
        self.decomposition = decomposition
        
    def fit(self, X):
        self.decomposition.fit(X)
        
    def transform(self, X):
        self.decomposition.transform(X)
    
    def fit_transform(self, X):
        self.decomposition.fit_transform(X)


# In[25]:


# 检查模型状态
def learning_curve_plot(estimator, X, y, train_sizes=np.linspace(0.05, 1., 20), cv=5, classifier=False, regressor=False, cluster=False, stdscaler=False, onehotencoder=False, labelencoder=False, vectorizer=False):
    estimator.fit(X, y, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
    X = estimator.features(X, stdscaler=stdscaler, onehotencoder=onehotencoder, labelencoder=labelencoder, vectorizer=vectorizer)
    if classifier:
        train_sizes, train_scores, validate_scores = learning_curve(estimator.classifier, X, y, train_sizes=train_sizes, cv=cv, n_jobs=-1)
    elif regressor:
        train_sizes, train_scores, validate_scores = learning_curve(estimator.regressor, X, y, train_sizes=train_sizes, cv=cv, n_jobs=-1)
    elif cluster:
        train_sizes, train_scores, validate_scores = learning_curve(estimator.cluster, X, y, train_sizes=train_sizes, cv=cv, n_jobs=-1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validate_scores_mean = np.mean(validate_scores, axis=1)
    validate_scores_std = np.std(validate_scores, axis=1)

    midpoint = ((train_scores_mean + train_scores_std) + (validate_scores_mean - validate_scores_std)) / 2
    diff = ((train_scores_mean + train_scores_std) - (validate_scores_mean - validate_scores_std))

    plt.figure(figsize=(16, 8))

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, color='b', alpha=0.1)
    plt.fill_between(train_sizes, validate_scores_mean - validate_scores_std,
                     validate_scores_mean + validate_scores_std, color='r', alpha=0.1)

    plt.plot(train_sizes, train_scores_mean, 'o-', color='b')
    plt.plot(train_sizes, validate_scores_mean, 'o-', color='r')
    plt.plot(train_sizes, midpoint, 'o-', color='k')

    plt.xlabel('样本数', fontproperties=font)
    plt.ylabel('精度', fontproperties=font)
    plt.legend(['训练集上的得分', '验证集上的得分', '期望得分'], loc='best', prop=font)
    plt.title('学习曲线', fontproperties=font)
    plt.grid(linestyle='dashed', alpha=0.2)
    print('训练集得分：', train_scores_mean[-1])
    print('验证集得分：', validate_scores_mean[-1])
    print('期望得分：', midpoint[-1])
    print('模型误差：', diff[-1])

sentences = []
def preprocess_text(content, sentences, category):
        for line in content:
            try:
                segs = jieba.lcut(line)
                segs = filter(lambda x: len(x) > 1, segs)
                segs = filter(lambda x: x not in stopword, segs)
                sentences.append((" ".join(segs), category))
            except Exception:
                print(line)
                continue