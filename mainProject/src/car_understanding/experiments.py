'''
Created on Jan 13, 2014

@author: ymovshov
'''

# import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pymc as mc
import time
import pandas as pd

import fgcomp_dataset_utils as fgu
from configuration import get_config
from attribute_selector import AttributeSelector
from attribute_classifier import AttributeClassifier
from bayes_net import BayesNet


def test_fg_utils():
  domain_meta_file = '/usr0/home/ymovshov/Documents/Research/Code/3rd_Party/fgcomp2013/release/domain_meta.txt'
  class_meta_file = '/usr0/home/ymovshov/Documents/Research/Code/3rd_Party/fgcomp2013/release/class_meta.txt'
  train_ann_file = '/usr0/home/ymovshov/Documents/Research/Code/3rd_Party/fgcomp2013/release/dataset.txt'

  print 'before load'
  domain_meta = fgu.read_domain_meta(domain_meta_file)
  class_meta = fgu.read_class_meta(class_meta_file)
  dataset = fgu.read_training_data(train_ann_file)
  print 'after load'

  print domain_meta.dtypes
  print '-------------'
  print class_meta.dtypes
  print '-------------'
  print dataset.dtypes

  print 'Showing the first two lines of dataset'
  print '------------------------------------------'
  print dataset.iloc[0:2,:]

  print ' '
  print 'Trying single function to load all data'
  print '------------------------------------------'
  config = get_config(small=True)
  (dataset, class_meta, domain_meta) = fgu.get_all_metadata(config)
  print dataset.iloc[0:2,:]


def load_hist_from_files(files, config=None):
  features = np.empty(shape=[0, config.SIFT.BoW.num_clusters])
  for curr_file in files:
    hist = Bow.load(file)
    features = np.concatenate([features, hist])

  return features

def dbg_clustering():
  (dataset, config) = small_run.preprocess()
  (features, labels) = small_run.create_feature_matrix(dataset, config)
#   fig, ax = plt.subplots()
#   heatmap = ax.pcolor(features, cmap=plt.cm.Blues)
#   plt.show()

#   imgplot = plt.imshow(features)
#   imgplot.set_cmap('gist_earth')
#   plt.colorbar()
#   plt.show()



def my_f(x):
  return x*x

def multi_test():
  from multiprocessing import Pool
  p = Pool(12)
  a = p.map(my_f, range(100))
  print a
  p.terminate()

# Define the Alarm node, which has B/E as parents
# @mc.deterministic(dtype=int)
# def Alarm(value=0, B=1, E=1):
#     """Probability of alarm given B/E"""
#     p = -np.Inf
#     if B and E:
#       p = 0.95
#     if B and not E:
#       p = 0.94
#     if E and not B:
#       p = 0.29
#     if not B and not E:
#       p = 0.001
#       
#     return -np.log(p)

def bayes_net_test():
  # trying the earthquake example from norvig
  
  # define the head nodes B and E
  B = mc.Bernoulli('B', p=0.001)
  E = mc.Bernoulli('E', p=0.002)
  
  
  # define the probability function for alaram
  def f_alarm(value=0, B=B, E=E):
    """Probability of alarm given B/E"""
    p = -np.Inf
    if B and E:
      p = 0.95
    if B and not E:
      p = 0.94
    if E and not B:
      p = 0.29
    if not B and not E:
      p = 0.001
      
    return p
  
  # define the alaram node (using the probability function)
  p_a = mc.Lambda('p_a', f_alarm)
  A = mc.Bernoulli('A', p_a)
  
  
  
  def f_john(value=0, A=A):
    """Probability of john given A"""
    if A:
      return 0.9
    return 0.05
  
  def f_mary(value=0, A=A):
    """Probability of john given A"""
    if A:
      return 0.7
    return 0.01
  
  p_j = mc.Lambda('p_j', f_john)
  J = mc.Bernoulli('J', p_j, value=True, observed=True)
  p_m = mc.Lambda('p_m', f_mary)
  M = mc.Bernoulli('M', p_m, value=True, observed=True)
  
  
  
  model = mc.Model([A, B, E, J, M])
  mc.graph.dag(model).write_pdf('tmp.pdf')

  MAP = mc.MAP(model)
  MAP.fit(method = 'fmin') # first do MAP estimation
  
  
  mcmc = mc.MCMC(model)
  mcmc.sample(100)
  
  
  mcmc.summary()
  
  b_samples = mcmc.trace('B')[:]
  print b_samples.shape
  print b_samples.mean()

  # plot stuff
#   plt.hist(b_samples, histtype='stepfilled', bins=10, alpha=0.85,
#          label="posterior of $B$", color="#A60628", normed=True)
#   plt.show()
#    
#    
#    
#   raw_input('press return when done')
  
  
  
  
  
#   print A.get_parents()



def class_ids_from_name(name, class_meta):
  pos_ids = []
  pos_name = str.lower(name)
  for ii in range(len(class_meta)):
    class_name = str.lower(class_meta['class_name'].iloc[ii])
    if str.find(class_name, pos_name) != -1:
      pos_ids.append(class_meta['class_index'].iloc[ii])

  return pos_ids
    
def select_small_set_for_bayes_net(dataset, makes, types):
  classes = dataset['class_meta']

  make_ids = set([])
  for make in makes:
    ids = class_ids_from_name(make, classes)
    make_ids.update(ids)

  c2 = classes[np.array(classes.class_index.isin(list(make_ids)))]
  final_ids = set([])
  for car_type in types:
    ids = class_ids_from_name(car_type, c2)
    final_ids.update(ids)

  c2 = c2[np.array(c2.class_index.isin(list(final_ids)))]
  return c2.copy()    

def classes_for_attribs():
  makes = ['bmw', 'ford']
  types = ['sedan', 'SUV']
  args = makes + types
  config = get_config(args)
  (dataset, config) = fgu.get_all_metadata(config)
  classes = select_small_set_for_bayes_net(dataset, makes, types)
  
  attrib_meta = pd.DataFrame(np.zeros([classes.shape[0], len(args)],dtype=int), 
                             columns = args,
                             index = classes.index)
  
  for class_index in attrib_meta.index:
    class_name = classes.class_name[class_index]
    for name in attrib_meta.columns:
      attrib_meta.ix[class_index, name] = \
      AttributeSelector.has_attribute_by_name(class_name, name)
       

  
  
  
  classes.to_csv('classes.csv')
  attrib_meta.to_csv('attribs.csv')
  

def cv_for_params():
  from sklearn.externals.joblib import load
  from sklearn.grid_search import GridSearchCV
  from sklearn.svm import SVC
  
  (X, y) = load('features_eq.tmp')
  
  # Set the parameters by cross-validation
  tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                       'C': [1, 10, 100, 1000]},
                      {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
  
  print("# Tuning hyper-parameters")
  print()

  clf = GridSearchCV(SVC(C=1,cache_size=2000, tol=1e-2), tuned_parameters, cv=5, scoring='precision',
                     n_jobs=-2, verbose=3)
  clf.fit(X, y)

  print("Best parameters set found on development set:")
  print()
  print(clf.best_estimator_)
  print()
  print("Grid scores on development set:")
  print()
  for params, mean_score, scores in clf.grid_scores_:
      print("%0.3f (+/-%0.03f) for %r"
            % (mean_score, scores.std() / 2, params))
  print()


def precision_recall():
#   from sklearn.metrics import roc_auc_score
#   from sklearn.metrics import roc_curve
  from sklearn.metrics import precision_recall_curve
  from sklearn.metrics import auc
  from sklearn.metrics import classification_report
  from mpltools import style
  style.use('ggplot')
  
  makes = ['bmw', 'ford']
  types = ['sedan', 'SUV']
  args = makes + types
  config = get_config(args)
  (dataset, config) = fgu.get_all_metadata(config)
  
  
  for ii, attrib_name in enumerate(args):
  #   attrib_name = 'bmw'
    
    attrib_clf = AttributeClassifier.load('../../../attribute_classifiers/{}.dat'.format(attrib_name))
    bnet = BayesNet(config, dataset['train_annos'], 
                    dataset['class_meta'], [attrib_clf], desc=str(args))
    
    res = bnet.create_attrib_res_on_images()
    
    attrib_selector = AttributeSelector(config, dataset['class_meta'])
  #   attrib_meta = attrib_selector.create_attrib_meta([attrib_clf.name])
    pos_classes = attrib_selector.class_ids_for_attribute(attrib_name)
    true_labels = np.array(res.class_index.isin(pos_classes))
    
   
    print "--------------{}-------------".format(attrib_name) 
    print res[str.lower(attrib_name)].describe()
    
    print classification_report(true_labels, np.array(res[str.lower(attrib_name)]) > 0.65, 
                                target_names=['not-{}'.format(attrib_name),
                                              attrib_name])
  
  
  
    precision, recall, thresholds = precision_recall_curve(true_labels, np.array(res[str.lower(attrib_name)]))
    score = auc(recall, precision)
    print("Area Under Curve: %0.2f" % score)
#     score = roc_auc_score(true_labels, np.array(res[str.lower(attrib_name)]))
#     fpr, tpr, thresholds = roc_curve(true_labels, np.array(res[str.lower(attrib_name)]))
    plt.subplot(2,2,ii+1)
#     plt.plot(fpr, tpr)
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.title('Precision-Recall: {}'.format(attrib_name))
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(['area = {}'.format(score)])
    
  plt.draw()
  plt.show()
  


def classify_using_attributes():
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import classification_report
  from sklearn import cross_validation
  
  
  makes = ['bmw', 'ford']
  types = ['sedan', 'SUV']
  args = makes + types
  config = get_config(args)
  (dataset, config) = fgu.get_all_metadata(config)
  
  attrib_names = [str.lower(a) for a in args]
  attrib_classifiers = []
  for attrib_name in args:
    attrib_classifiers.append(AttributeClassifier.load('../../../attribute_classifiers/{}.dat'.format(attrib_name)))
  
  classes = select_small_set_for_bayes_net(dataset, makes, types)
  train_annos = dataset['train_annos']
  train_annos = train_annos[np.array(
                             train_annos.class_index.isin(classes.class_index))]
  
  bnet = BayesNet(config, train_annos, 
                  classes, attrib_classifiers, desc=str(args))
  
  res = bnet.create_attrib_res_on_images()
  
  # define a classifier that uses the attribute scores
  clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
  
  scores = cross_validation.cross_val_score(clf, res[attrib_names], 
                                            res.class_index, cv=5)
  print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#   clf.fit(res[attrib_names], res.class_index)
  
  y_pred = np.array(clf.predict(res[attrib_names]))
  y_true = np.array(res.class_index)
  
  print(classification_report(y_true, y_pred))
    
  

if __name__ == '__main__':
#   test_fg_utils()
#   dbg_clustering()
#     test_work_remote
#   multi_test()
#   bayes_net_test()
#   classes_for_attribs()
#   cv_for_params()
#   precision_recall()
#   bayes_net_test()
  classify_using_attributes()