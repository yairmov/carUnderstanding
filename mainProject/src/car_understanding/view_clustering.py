'''
Created on Mar 25, 2014

@author: ymovshov
'''
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


def cluster(data_anno, config, K):
  return cluster_aspect_ration(data_anno, config, K)

def cluster_aspect_ration(data_anno, config, K):
  ar = pd.DataFrame((data_anno.xmax - data_anno.xmin) / (data_anno.ymax - data_anno.ymin),
                    columns=['ar'])
  ar_model = KMeans(n_clusters=K, verbose=False)
  aa = np.array(ar)
  if len(aa.shape) == 1:
      aa.reshape([aa.shape[0], 1])
  labels = pd.Series(data=ar_model.fit_predict(aa), index=ar.index)
  
  return labels
