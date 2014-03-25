'''
Created on Mar 25, 2014

@author: ymovshov
'''
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


def cluster(train_annos, config, K):
  return cluster_aspect_ration(train_annos, config, K)

def cluster_aspect_ration(train_annos, config, K):
  ar = pd.DataFrame((train_annos.xmax - train_annos.xmin) / (train_annos.ymax - train_annos.ymin),
                    columns=['ar'])
  ar_model = KMeans(n_clusters=K, verbose=False)
  aa = np.array(ar)
  if len(aa.shape) == 1:
      aa.reshape([aa.shape[0], 1])
  labels = pd.Series(data=ar_model.fit_predict(aa), index=ar.index)
  
  return labels
