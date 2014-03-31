'''
Created on Jan 16, 2014

@author: ymovshov
'''

import os as os
# import pandas as pd
import numpy as np
from sklearn import cross_validation
# from sklearn import svm
from sklearn import preprocessing
from sklearn.externals.joblib import Parallel, delayed, dump, load
import cv2 as cv
import pandas as pd
import collections
import matplotlib.pyplot as plt

from configuration import get_config
import fgcomp_dataset_utils as fgu
from dense_SIFT import dense_SIFT, save_to_disk, load_from_disk
import Bow as Bow
from attribute_classifier import AttributeClassifier
from bayes_net import BayesNet
from util import ProgressBar

# def preprocess(args):
#   config = get_config(args)
#   (train_annos, class_meta, domain_meta) = fgu.get_all_metadata(config)
# 
#   # Filter the class meta and train annotations according to the small use
#   # case definitions
#   class_meta = class_meta[class_meta['domain_index'] == config.dataset.domains[0]]
#   train_annos = train_annos[train_annos.class_index.isin(class_meta.class_index)]
# 
#   return ({'train_annos': train_annos,
#              'class_meta': class_meta,
#              'domain_meta': domain_meta},
#           config)


def class_ids_from_name(name, class_meta):
  pos_ids = []
  pos_name = str.lower(name)
  for ii in range(len(class_meta)):
    class_name = str.lower(class_meta['class_name'].iloc[ii])
    if str.find(class_name, pos_name) != -1:
      pos_ids.append(class_meta['class_index'].iloc[ii])

  return pos_ids


def img_ids_for_classes(class_ids, train_annos):
  return train_annos[train_annos.class_index.isin(class_ids)].index



def calc_dense_SIFT_one_img(annotation, config):
  rel_path = annotation['rel_path']
  img_file = os.path.join(config.dataset.main_path, rel_path)

  # Replace extension to .dat and location in config.SIFT.raw_dir
  (name, ext) = os.path.splitext(os.path.split(img_file)[1])
  save_name = os.path.join(config.SIFT.raw_dir, name + '.dat')

  if os.path.exists(save_name):
    return

  img = cv.imread(img_file)
  (kp, desc) = dense_SIFT(img, grid_spacing=config.SIFT.grid_spacing)
  save_to_disk(kp, desc, save_name)


def calc_dense_SIFT_on_dataset(dataset, config):
  '''
  Just calls calc_dense_SIFT_one_img on all images in dataset using a
  parallel wrapper.
  '''
#   train_annos = dataset['train_annos']

  Parallel(n_jobs=-1, verbose=config.logging.verbose)(
                 delayed(calc_dense_SIFT_one_img)(dataset.iloc[ii], config)
                 for ii in range(len(dataset)))

def contains(box, point):
  '''
  box = (xmin, xmax, ymin, ymax)
  point = (x, y)
  '''
  return (box[0] <= point[0] and box[1] >= point[0] and
          box[2] <= point[1] and box[3] >= point[1])



def load_SIFT_from_a_file(curr_anno, config):
  curr_file = os.path.splitext(curr_anno['basename'])[0] + '.dat'
  (kp, desc) = load_from_disk(os.path.join(config.SIFT.raw_dir, curr_file))

  # Only keep points that are inside the bounding box
  box = (curr_anno['xmin'], curr_anno['xmax'],
         curr_anno['ymin'], curr_anno['ymax'])

  inds = np.zeros(shape=[len(kp),], dtype=bool)
  for jj in range(len(kp)):
    inds[jj] = contains(box, kp[jj].pt)

  desc = desc[inds, :]
  kp = np.asarray(kp)[inds].tolist()

  # Random selection of a subset of the keypojnts/descriptors
  inds  = np.random.permutation(desc.shape[0])
  desc = desc[inds, :]
  desc = desc[:config.SIFT.BoW.max_desc_per_img, :]
#   kp    = [kp[i] for i in inds]
#   kp    = kp[:config.SIFT.BoW.max_desc_per_img]

  return desc

def load_SIFT_from_files(dataset, config):
  train_annos = dataset['train_annos']

  nfiles = len(train_annos)
  print 'Loading dense SIFT for %d training images ' % nfiles
  features = Parallel(n_jobs=-1, verbose=config.logging.verbose)(
                 delayed(load_SIFT_from_a_file)(train_annos.iloc[ii], config)
                 for ii in range(nfiles))

#   features = []
#   pbar = ProgressBar(nfiles)
#   for ii in range(nfiles):
#     pbar.animate(ii)
#     features.append(load_SIFT_from_a_file(train_annos.iloc[ii], config))

  # convert to numy arry
  features = np.concatenate(features)

  # sample max_desc features
  inds  = np.random.permutation(features.shape[0])
  features = features[inds, :]
  features = features[:config.SIFT.BoW.max_desc_total, :]

  return features

def create_feature_matrix(dataset, config):
  train_annos = dataset['train_annos']

  # Preallocate feature matrix
  features = np.empty(shape=[len(train_annos), config.SIFT.BoW.num_clusters])

  # Load histograms from disk into a matrix
  print 'Loading histograms from disk'
  for ii in range(len(train_annos)):
    img_name = train_annos.iloc[ii]['basename']
    img_name = os.path.splitext(img_name)[0]
    hist_filename = os.path.join(config.SIFT.BoW.hist_dir, img_name) + '_hist.dat'
    hist = Bow.load(hist_filename) # hist[0] = values, hist[1] = bin edges
    features[ii, :] = hist[0]


  # preprocess features
  features = preprocessing.scale(features)

  # create pos/neg labels
  print 'Creating pos/neg labels'
  labels = train_annos.class_index.isin(
                    config.dataset.class_ids.pos).values

  return (features, labels)

def evaluate(features, labels):
  from sklearn.cross_validation import train_test_split
  from sklearn.grid_search import GridSearchCV
  from sklearn.metrics import classification_report

#   # Split the train_annos in two parts
#   X_train, X_test, y_train, y_test = train_test_split(
#     features, labels, test_size=0.3, random_state=0)
#
#
#   # Set the parameters by cross-validation
#   tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                        'C': [1, 10, 100, 1000]},
#                       {'kernel': ['linear'], 'C': [0.005, 1, 10, 100, 1000]}]
#
#   scores = ['precision', 'recall']
#
#   for score in scores:
#       print("# Tuning hyper-parameters for %s" % score)
#       print()
#
#       clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5, scoring=score)
#       clf.fit(X_train, y_train)
#
#       print("Best parameters set found on development set:")
#       print()
#       print(clf.best_estimator_)
#       print()
#       print("Grid scores on development set:")
#       print()
#       for params, mean_score, scores in clf.grid_scores_:
#           print("%0.3f (+/-%0.03f) for %r"
#                 % (mean_score, scores.std() / 2, params))
#       print()
#
#       print("Detailed classification report:")
#       print()
#       print("The model is trained on the full development set.")
#       print("The scores are computed on the full evaluation set.")
#       print()
#       y_true, y_pred = y_test, clf.predict(X_test)
#       print(classification_report(y_true, y_pred))
#       print()


########

  # SVM params
  print 'setting classifier parameters'
#   clf = svm.SVC(kernel='linear', C=0.0005, class_weight='auto')
#   clf = svm.SVC(kernel='rbf', C=10, gamma=0.0001, class_weight='auto')
#   from sklearn.ensemble import AdaBoostClassifier
#   clf = AdaBoostClassifier(svm.SVC(kernel='linear', C=0.005),
#                            algorithm="SAMME",
#                          n_estimators=10)

  from sklearn.ensemble import RandomForestClassifier
  clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
  scores = cross_validation.cross_val_score(clf, features, labels, cv=5)

  # Report results
  print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
  return (clf, scores)


def print_output(clf, scores, config):
  if not os.path.isdir(config.output_dir):
    os.makedirs(config.output_dir)

  out_name = os.path.join(config.output_dir,
                          'classification_results_{}_{}.txt'.format(
                          config.attribute.pos_name, config.attribute.neg_name))

  with open(out_name, 'w') as f:
    f.write("Config:\n")
    f.write("----------\n")
    f.write(config.makeReport() + "\n")
    f.write("----------\n\n")


    f.write("{}\n".format(clf))
    f.write("Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))


def run_attrib_training(args, cross_validation=False):
#   (dataset, config) = preprocess(args)
  config = get_config(args)
  (dataset, config) = fgu.get_all_metadata(config)

  #  RUN dense SIFT on alll images
#   print "Saving Dense SIFT to disk"
#   calc_dense_SIFT_on_dataset(dataset['train_annos'], config)

  # Create BoW model
#   features = load_SIFT_from_files(dataset, config)
#   print "Loaded %d SIFT features from disk" % features.shape[0]
#   print "K-Means CLustering"
#   bow_model = Bow.create_BoW_model(features, config)
#   print("number of actual clusters found: {}".format(bow_model.n_clusters))
#   Bow.save(bow_model, config.SIFT.BoW.model_file)
#   return

  # Assign cluster labels to all images
  print "Assigning to histograms"
  Bow.create_word_histograms_on_dataset(dataset['real_train_annos'], config)
  return

  # Train attribute classifiers
  print "Training attribute classifiers"
  print "+++++++++++++++++++++++++++++++"
  for attrib_name in config.attribute.names:
    print attrib_name
    pos_class_ids = class_ids_from_name(attrib_name, dataset['class_meta'])
    pos_img_ids   = img_ids_for_classes(pos_class_ids,
                                                dataset['train_annos'])
    attrib_clf = AttributeClassifier(config,
                                     dataset['train_annos'],
                                     pos_img_ids,
                                     attrib_name,
                                     desc=attrib_name)

    attrib_clf.run_training_pipeline(cross_validation)
    
    AttributeClassifier.save(attrib_clf, os.path.join(config.attribute.dir,
                                                      attrib_clf.name + '.dat'))


    print "-------------------------------------"
    print "-------------------------------------"




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



def create_attrib_res_on_images(train_annos, attrib_classifiers, config):
  print "Load image Bow histograms from disk"
  features = np.empty(shape=[len(train_annos), config.SIFT.BoW.num_clusters])
  for ii in range(len(train_annos)):
    img_name = train_annos.iloc[ii]['basename']
    img_name = os.path.splitext(img_name)[0]
    hist_filename = os.path.join(config.SIFT.BoW.hist_dir,
                                 img_name) + '_hist.dat'
    hist = Bow.load(hist_filename) # hist[0] = values, hist[1] = bin edges
    features[ii, :] = hist

  print "Apply attribute classifiers on images"
  res = {}
  for attrib_clf in attrib_classifiers:
    curr_res = attrib_clf.clf.decision_function(features)
    res[attrib_clf.name] = curr_res.reshape(len(curr_res))

  res = pd.DataFrame(data=res, index=train_annos.index)


  res = pd.concat([res, train_annos.ix[:, ['class_index']]], axis=1)
  return res



def calc_stats(clf_res):
  mu = {}
  sig = {}
  for c in np.unique(clf_res.class_index):
    mu[c]  = clf_res[clf_res.class_index == c].mean().iloc[:4]
    sig[c] = clf_res[clf_res.class_index == c].std().iloc[:4]


  prob_c = collections.Counter(clf_res.class_index)
  s = sum(prob_c.values())
  for c in prob_c.keys():
    prob_c[c] /= float(s)
  return {'mu': mu, 'sig': sig, 'prob_c': prob_c}


def predict_using_bayes(clf_res, prob_c, mu, sig, dataset, config):
  attrib_names = mu[mu.keys()[0]].index
  class_names = dataset['class_meta'].class_name[prob_c.keys()]
  posteriors = pd.DataFrame(data=np.zeros(shape=[clf_res.shape[0],
                                                 len(prob_c.keys())]),
                            index=clf_res.index,
                            columns=class_names)

  gauss_norm = 1 / (np.sqrt(2 * np.pi))

  for ii in range(len(clf_res)):
    curr_res = clf_res.iloc[ii]
    for c in prob_c.keys():
      mu_c = mu[c]
      sig_c = sig[c]
      p_c = prob_c[c]
      # p(a|c) ~ N(mu_ac, sig_ac)
      for a_name in attrib_names:
        mu_ac = mu_c[a_name]
        sig_ac = sig_c[a_name]
        gauss_exp = (-1/(2 * sig_ac * sig_ac)) * np.power(
                                                curr_res[a_name] - mu_ac, 2)
        p_c *= (1/sig_ac)*gauss_norm * np.exp(gauss_exp)
        posteriors.iloc[ii][class_names[c]] = p_c



  # add true class as last column
  train_annos = dataset['train_annos']
  t = train_annos[train_annos.index.isin(clf_res.index)]
  posteriors = pd.concat([posteriors,
                          t.ix[:, ['class_name']]],
                         axis=1)
  posteriors.rename(columns={'class_name': 'true_class'}, inplace=True)

  posteriors['predicted'] = posteriors.ix[:,:-1].idxmax(axis=1)

  # shorten col names
  posteriors.rename(columns=lambda x: x[:-5], inplace=True)
  posteriors.rename(columns={'true_':'tr', 'pred':'pr'}, inplace=True)
  posteriors['success'] = posteriors.tr == posteriors.pr

  return posteriors

def bayes_net_generic(use_gt=False):
#   makes = ['bmw', 'ford']
#   types = ['sedan', 'SUV']
#   args = makes + types
  
  with open('sorted_attrib_list.txt', 'r') as f:
    args = f.readlines()
  args = [x.strip() for x in args]  
  
  # use only top K
  K = 36
  args = args[:K]
  
  config = get_config(args)
  (dataset, config) = fgu.get_all_metadata(config)
  
#   print "training attrib classifiers"
#   run_attrib_training(args, cross_validation=False)
#   print "Returning after training attrib classifiers"
#   return
  
  attrib_classifiers = []
  if use_gt:
    class dummy:
      def __init__(self, name):
        self.name = name
    
    for name in args:
      attrib_classifiers.append(dummy(name))
  else:
    for name in args:
      filename = os.path.join(config.attribute.dir, name + '.dat')
      attrib_classifiers.append(AttributeClassifier.load(filename))
  
  train_annos = dataset['train_annos']  
  # Select only images from the args "world"
  classes = dataset['class_meta']
#   classes = select_small_set_for_bayes_net(dataset, makes, types)
#   train_annos = train_annos[np.array(
#                              train_annos.class_index.isin(classes.class_index))]
  
  bnet = BayesNet(config, train_annos, 
                  classes, attrib_classifiers, 
                  desc=str(args), use_gt=use_gt)
  bnet.init_CPT()
  
  
  test_annos = dataset['test_annos']
  # Select only images from the args "world"
#   test_annos = test_annos[np.array(
#                              test_annos.class_index.isin(classes.class_index))]
  
  
  (class_probs, attrib_probs) = bnet.predict(test_annos)
  show_confusion_matrix(test_annos, classes, class_probs)
  dump({'class_probs': class_probs, 'attrib_probs': attrib_probs},
       'bnet_res.dat')
  
  
  
def show_confusion_matrix(train_annos, class_meta, class_prob):
  from sklearn.metrics import confusion_matrix
  from sklearn.preprocessing import normalize
  from mpltools import style
  from sklearn.metrics import classification_report
  from sklearn.metrics import accuracy_score
  
#   style.use('ggplot')
  
  class_true = train_annos.class_index
  y_pred_class  = class_prob.idxmax(axis=1)
  
#   y_pred_attrib = attrib_prob.idxmax(axis=1)
  
  class_inds = np.sort(np.array(class_prob.columns))
  class_names = class_meta.class_name[class_inds]
  cc = class_names.apply(lambda x: str.find(str.lower(x), 'sedan') != -1)
  cc.sort()
  class_names = class_names[cc.index]
  print classification_report(class_true, y_pred_class, list(class_inds), list(class_names))
  print "Accuracy: {}".format(accuracy_score(class_true, y_pred_class))
  cm = confusion_matrix(class_true, y_pred_class, class_names.index) + 0.0
  cm = normalize(cm, norm='l1', axis=1)
  
  
  # Show confusion matrix in a separate window
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(111)
  cax = ax.matshow(cm)
  plt.title('Confusion matrix of the classifier')
  fig.colorbar(cax)
  ax.set_xticks(np.arange(len(class_names)))
  ax.set_xticklabels([''] + np.array(class_names), rotation=45, ha='left')
  ax.set_yticks(np.arange(len(class_names)))
  ax.set_yticklabels([''] + np.array(class_names))
  plt.xlabel('Predicted')
  plt.ylabel('True')
#   fig.savefig('cm.pdf')
  plt.show() 



if __name__ == '__main__':

#   args = ["sedan", "SUV", "2012", "Audi", "bmw", "ford",
#           "chevrolet", "coupe", "hatchback", "dodge", "hyundai"]
  
#   args = ["sedan", "SUV", "bmw", "ford"]
  args = ["sedan"]

  run_attrib_training(args, cross_validation=True) 

#   # Small Bayes net (naive bayes...)
#   makes = ['bmw', 'ford']
#   types = ['sedan', 'SUV']
#   args = makes + types
#   (dataset, config) = preprocess(args)
# 
#   # Select only images from the args "world"
#   classes = select_small_set_for_bayes_net(dataset, makes, types)
#   train_annos = dataset['train_annos']
#   train_annos = train_annos[np.array(
#                              train_annos.class_index.isin(classes.class_index))]
# 
#   attrib_classifiers = []
#   for name in args:
#     filename = os.path.join(config.attribute.dir, name + '.dat')
#     attrib_classifiers.append(AttributeClassifier.load(filename))
# 
#   clf_res = create_attrib_res_on_images(train_annos, attrib_classifiers, config)
# 
# 
#   print "clf_res\n-------"
#   print clf_res.head()
# 
#   stats = calc_stats(clf_res)
# 
#   dump({'clf_res': clf_res, 'stats':stats}, 'tmp.dat')
# 
#   post = predict_using_bayes(clf_res, stats['prob_c'], stats['mu'], stats['sig'],
#                       dataset, config)
# 
#   print "Accuracy = {}".format(sum(post.success) / float(len(post)))

  
  # Using the more generic BayesNet class
  #-------------------------------------
  
#   bayes_net_generic(use_gt=True)











#   makes = ['bmw', 'ford']
#   types = ['sedan', 'SUV']
#   args = makes + types
# #   (dataset, config) = preprocess(args)
#   config = get_config(args)
#   (dataset, config) = fgu.get_all_metadata(config)  
# 
#   classes = select_small_set_for_bayes_net(dataset, makes, types)
#   train_annos = dataset['train_annos']
#   train_annos = train_annos[np.array(
#                              train_annos.class_index.isin(classes.class_index))]
#   
#   a = load('bnet_res.dat')
#   class_prob = a['class_probs']
#   attrib_prob = a['attrib_probs']
#   dump((train_annos, classes, class_prob, attrib_prob), 'tmp.dat')
#   show_confusion_matrix(train_annos, classes, class_prob)






