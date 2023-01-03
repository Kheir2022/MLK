#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing libraries
import os
import random
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import ensemble,neural_network,neighbors,svm,model_selection,inspection
warnings.simplefilter(action='ignore')


# In[ ]:


# Functions
def performanceStatistics(x, y):
  E = y - x                       # Error
  AE = np.abs(E)                  # Absolute Error
  MAE = np.mean(AE)               # Mean Absolute Error
  SE = np.power(E, 2)             # Square Error
  MSE = np.mean(SE)               # Mean Square Error (this method is the best)
  RMSE = np.sqrt(MSE)             # Root Mean Square Error
  RB = ((np.mean(y) - np.mean(x)) / np.mean(x)) * 100
  slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
  R2 = np.power(r_value, 2)       # correlation of determination
  stat = {'R2':round(R2, 2),
          'RB':round(RB, 2),
          'MAE':round(MAE, 2),
          'RMSE':round(RMSE, 2),
          'n':len(E)}
  return stat

def plotOriginalPredicted(Original, Predicted, outFile, label=''):
  minX = np.min(Original)
  maxX = np.max(Original)
  minY = np.min(Predicted)
  maxY = np.max(Predicted)
  minXY = np.min(np.array([minX, minY]))
  maxXY = np.min(np.array([maxX, maxY]))
  fs = 26
  fig = plt.figure(figsize=(15, 15))
  plt.scatter(Original, Predicted, color='blue', label=label)
  plt.plot(np.linspace(minXY, maxXY, 50), np.linspace(minXY, maxXY, 50), color='red', linestyle='-', linewidth=1, markersize=5, label='1:1 Line')
  plt.xlim([minXY, maxXY])
  plt.ylim([minXY, maxXY])
  plt.xticks(size = fs)
  plt.yticks(size = fs)
  plt.xlabel('Original yield (kg/ha)', fontsize=fs)
  plt.ylabel('Predicted yield (kg/ha)', fontsize=fs)
  
  stat = performanceStatistics(Original, Predicted)
  digits = 2
  n = round(stat['n'], digits)
  r2 = round(stat['R2'], digits)
  RB = round(stat['RB'], digits)
  MAE = round(stat['MAE'], digits)
  RMSE = round(stat['RMSE'], digits)
  s = 'n={} \n$R^2$={}\nRB={} (%)\nMAE={} (kg/ha)\nRMSE={} (kg/ha)'.format(n, r2, RB, MAE, RMSE)
  plt.text(x=(minXY + 100), y=(maxXY - 100), s=s, horizontalalignment='left', verticalalignment='top', color='black', fontsize=fs)
  plt.legend(loc=9, fontsize=fs)
  plt.savefig(outFile, dpi=300, bbox_inches='tight')
  plt.clf()
  plt.close()

def hyperparameters(n):
  
  ANN_param_grid = dict(hidden_layer_sizes = hls , 
                        activation = ['identity', 'logistic', 'tanh', 'relu'], # 'identity', 'logistic', 'tanh', 'relu'
                        solver = ['lbfgs', 'sgd', 'adam'], # 'lbfgs', 'sgd', 'adam'
                        max_iter = [100000],
                        )
  
   
  MLA = {

      # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
      'ANN':[neural_network.MLPRegressor(random_state=0, verbose=False), ANN_param_grid],

      }

  return MLA
# Features
xFeatures = ['Years','Adate','Mdate','Biomass','HWUM','H#AM','LAIx','ETCM','WPET','Wpirri','NDCH','TMAXA',
             'TMINA','SRADA','DAYLA','PRCP','Clay','BD']

yFeatur = 'Yield (kg/ha)'


# In[ ]:


# Work space
workspace = '/content/drive/My Drive/EIA'
dataFile  = os.path.join(workspace, 'PotentialMME.xlsx')


# In[ ]:


#Traning and testing ML

df = pd.read_excel('/content/drive/My Drive/EIA/PotentialMME.xlsx')
Xdf = df[xFeatures]
ydf = df[[yFeatur]]

X = df[xFeatures]
y = df[[yFeatur]]
X = (X - Xdf.mean()) / Xdf.std()
y = (y - ydf.mean()) / ydf.std()
df = pd.concat([X, y], axis = 1)
df = df.dropna()
X = df[xFeatures]
y = df[[yFeatur]]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)

n = len(xFeatures)
MLA = hyperparameters(n)
for idMLA, mlaL_mlaA in enumerate(MLA.items()):
  mlaL, mlaA = mlaL_mlaA
  estimator = mlaA[0]
  param_grid = mlaA[1]
  GSCV = model_selection.GridSearchCV(estimator, param_grid, scoring='r2', cv=5, refit=True, verbose=False, n_jobs=-1)
  GSCV.fit(X_train, y_train)
  bestEstimator = GSCV.best_estimator_ 

  OriginalYield_test  = y_test.values.flatten()
  PredictedYield_test = bestEstimator.predict(X_test).flatten()
  OriginalYield_test  = (float(ydf.std()) * OriginalYield_test) + float(ydf.mean())
  PredictedYield_test = (float(ydf.std()) * PredictedYield_test) + float(ydf.mean())
  
  outFile     = os.path.join(workspace, '{}_plot_DefaultML_{}.png'.format(mlaL, n))
  Stats       = os.path.join(workspace, '{}_Stats_DefaultML_{}.xlsx'.format(mlaL, n))
  Importances = os.path.join(workspace, '{}_Importances_DefaultML_{}.xlsx'.format(mlaL, n))

  plotOriginalPredicted(OriginalYield_test, PredictedYield_test, outFile, label='Predicted yield of {}'.format(mlaL))
  df_stats = performanceStatistics(OriginalYield_test, PredictedYield_test)
  print(mlaL, df_stats)

  df_GSCV = pd.DataFrame(GSCV.cv_results_)
  df_GSCV = df_GSCV.sort_values(by='rank_test_score', ascending=True)
  df_GSCV = df_GSCV.head(1)
  for i in list(df_GSCV.columns):
    df_stats[i] = df_GSCV[i].values.tolist()[0]
  df_stats = pd.DataFrame.from_dict(df_stats, orient='index').T
  df_stats.to_excel(Stats, index=False)
  
  pi = inspection.permutation_importance(bestEstimator, X_train, y_train, n_jobs=-1, random_state=0).importances_mean
  pi = [((i / pi.sum()) * 100) for i in pi]
  dfImportances = pd.DataFrame(data=[pi], columns=xFeatures).round(2)
  dfImportances.to_excel(Importances, index=False)

