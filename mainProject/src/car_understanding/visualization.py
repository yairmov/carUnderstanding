'''
Created on Mar 31, 2014

@author: ymovshov
'''

from __future__ import print_function
import base64
import numpy as np
from sklearn import ensemble, decomposition, manifold
from PIL import Image
from path import path
import pandas as pd
import matplotlib.pyplot as plt
import scipy.misc
import cv2 as cv
import os

import Bow
from view_clustering import cluster
from attribute_selector import AttributeSelector
from attribute_classifier import AttributeClassifier

# boxes should be a tuple/list of tuples. each inner tuple should be
# of format (xmin, xmax, ymin, ymax)
def showboxes(img_in, boxes, is_opencv=False):

  if is_opencv:
    img = cv.cvtColor(img_in, cv.COLOR_BGR2RGB)
  else:
    img = img_in

  ax = plt.axes()
  ax.imshow(img)
  ax.axis('off')

  for box in boxes:
    xmin, xmax = box[0], box[1]
    ymin, ymax = box[2], box[3]
    r = plt.Rectangle((xmin, ymin),
                   xmax-xmin, ymax-ymin,
                     edgecolor='red', facecolor='none')
    ax.add_patch(r)

  plt.draw()
  plt.show()


def explore_image_data(annos, config):
  plt.ion()
  n_imgs = len(annos)
  for ii in reversed(range(n_imgs)):
    curr_anno = annos.iloc[ii]
    img = scipy.misc.imread(curr_anno['img_path'])
    bbox = (curr_anno['xmin'], curr_anno['xmax'],
            curr_anno['ymin'], curr_anno['ymax'])
    print('BB width: %g' % (bbox[1] - bbox[0]))
    boxes = (bbox,) # create a tuple, as it is expected by showboxes
    showboxes(img, boxes, is_opencv=False)
    c = raw_input('Press any key to continue\n')
    plt.clf()
    if c.startswith('q'):
      plt.close()
      return


  plt.close()


def show_best_predictions(test_annos, attrib_names, html_file, config):
  topK = 5
  
  print("Load image Bow histograms from disk")
  features = Bow.load_bow(test_annos, config)
  
  attrib_clfs = []
  for name in attrib_names:
    attrib_clfs.append(AttributeClassifier.load('../../../attribute_classifiers/{}.dat'.format(name)))
    
    
  img_files = []
  names = []  
  for attrib_clf in attrib_clfs:
    print(attrib_clf.name)
    names.append(attrib_clf.name)
    res = pd.DataFrame(data=attrib_clf.decision_function(features), 
                       index=test_annos.index)
    inds = np.array(res.sort([0], ascending=False).index[:topK])
    img_files.extend(list(test_annos.loc[inds].img_path))


  create_image_page(img_files, html_file, num_per_row=topK, usr_str = names)
    
  


def create_image_page(img_files, html_file, width=200, num_per_row=9,
                                  split_every=np.Inf, usr_str=None):

  k = 0;
  split_num = 0
  group = 0

  width_str = 'width = ' + str(width)
  html_code_image = '<img src="data:image/jpeg;base64, {}" ' + width_str + ' style="border:1px solid white" >'

  html_str = '<html><body> '

  # Create HTML string
#   html_str += '<center> <h2> ' + str(split_num) + '</h2></center> <br>'
  for f in img_files:
    with open(f, "rb") as img_file:
      img_str_64 = base64.b64encode(img_file.read())

#     img_name = f
    s = html_code_image.format(img_str_64)
    html_str += s + '\n'
    k += 1

    if (k % split_every == 0):
      html_str += '<hr>' + '<center> <h2> ' + str(split_num) + '</h2></center> <br> <hr>'
      split_num += 1

    if (k == num_per_row):
      if not(usr_str is None):
        html_str += '<bold>' + usr_str[group] + '</bold><br> <hr>'
#         html_str += '<hr>' + '<center> <h2> ' + usr_str[group] + '</h2></center> <br> <hr>'
        group += 1
      k = 0
#       html_str += '<hr>'
#       html_str += str(split_num) + '<br> <hr>'
      split_num += 1


  html_str += '</body></html>'

  # Write to file
  with open(html_file, "w") as out_file:
    out_file.write(html_str)




def plot_dataset_embedding(data_annos, config,
                           labels=None, title=None,
                           show_images=True):
  '''
  Display a figure that shows an 2D embedding of the BoW features for the
  data_annos. It also loads the images of the data, and displays them on the
  figure (when there is enough space)
  '''
  n_items = data_annos.shape[0]
  print('Loading {} BoW from disk'.format(n_items))
  features = Bow.load_bow(data_annos, config)

  if labels is None:
    labels = data_annos.class_index


  # -- randomd tree embedding
  hasher = ensemble.RandomTreesEmbedding(n_estimators=400, random_state=0,
                                       max_depth=5)
  X_transformed = hasher.fit_transform(features)
  pca = decomposition.TruncatedSVD(n_components=2)
  X_reduced = pca.fit_transform(X_transformed)

  # LLE
#   clf = manifold.LocallyLinearEmbedding(30, n_components=2,
#                                       method='standard')
#   X_reduced = clf.fit_transform(features)

  # read images form disk
  images = None
  if show_images:
    p = path(config.dataset.main_path)
    img_names = data_annos.img_path
    images = []
    for ii in range(len(data_annos)):
      img_name = img_names.iloc[ii]
      im = Image.open(img_name)
      im.thumbnail([50,50])
      images.append(np.array(im))

  plot_embedding(X_reduced, y=labels, images=images, title=title)

#----------------------------------------------------------------------
# Scale and visualize embedding vectors
# X - the vectors to plot
# y - Labels
def plot_embedding(X, y=None, images=None, title=None):
  import pylab as pl
  from matplotlib import offsetbox


  x_min, x_max = np.min(X, 0), np.max(X, 0)
  X = (X - x_min) / (x_max - x_min)
  if y is None:
    y = np.ones(shape=[X.shape[0], ])

  pl.figure(figsize=[20,20])
  ax = pl.subplot(111)

  if (not images is None) and hasattr(offsetbox, 'AnnotationBbox'):
    shown_images = np.array([[1., 1.]])  # just something big
    for i in range(X.shape[0]):
      dist = np.sum((X[i] - shown_images) ** 2, 1)
      if np.min(dist) < 4e-3:
          # don't show points that are too close
        continue
      shown_images = np.r_[shown_images, [X[i]]]
      imagebox = offsetbox.AnnotationBbox(
          offsetbox.OffsetImage(images[i], cmap=pl.cm.gray_r),
          X[i])
      ax.add_artist(imagebox)

  labels = np.unique(y)
  m = labels.min()
  pl.scatter(X[:,0], X[:,1], s=80,
          c=y-m / float(len(labels)),
          marker='o', cmap=pl.cm.Set1, alpha=0.6, linewidths=1)
#   for i in range(len(y)):
#     pl.text(X[i, 0], X[i, 1], str(y.iloc[i]),
#             color=pl.cm.Set1(float(y.iloc[i] - m) / len(labels)),
#             fontdict={'weight': 'bold', 'size': 9})

  pl.xticks([]), pl.yticks([])
  if title is not None:
    pl.title(title)
    
    
    
def show_feature_matrix(annos, config, class_meta, attrib_names):
  features = Bow.load_bow(annos, config)
  features = pd.DataFrame(data=features, index=annos.index)
  
  K = 16
  labels = cluster(annos, config, K)
  # use cluster with median size
  c = np.bincount(labels)
  median_label = np.where(c == np.sort(c)[int(np.floor(c.shape[0] / 2))])[0][0]
  d = annos[labels == median_label]
#   curr_feat = features.loc[d.index]

  attrib_selector = AttributeSelector(config, class_meta)
  
  
  n_attrib = len(attrib_names)
  
  fig = plt.figure()
  figrows = figcols = np.ceil(np.sqrt(n_attrib))  
  for ii, attrib_name in enumerate(attrib_names):
    pos_class_ids = attrib_selector.class_ids_for_attribute(attrib_name)
    pos_img_ids = d[d.class_index.isin(pos_class_ids)].index
    y = pd.Series(data = False, index=d.index)
    y[pos_img_ids] = True
    y.sort(ascending=False)
    curr_feat = features.loc[y.index]
    plt.subplot(figrows, figcols, ii+1)
    
    # show the feature vectors
    plt.imshow(curr_feat); plt.axis('off'); plt.axis('normal'); plt.axis('tight')
    plt.title(attrib_name)
    
    # draw a line to seperate pos/neg
    n_pos = pos_img_ids.shape[0]
    plt.hlines(n_pos, 0, curr_feat.shape[1], 
               color='k', linestyle='dashed', 
               linewidth=2)
    
  plt.suptitle('Features from cluster: {}'.format(median_label))
  plt.show()
    
    
  
  def series_to_iplot(series, name=''):
    '''
    Coverting a Pandas Series to Plotly interface
    '''
    if series.index.__class__.__name__=="DatetimeIndex":
        #Convert the index to MySQL Datetime like strings
        x = series.index.format()
        #Alternatively, directly use x, since DateTime index is np.datetime64
        #see http://nbviewer.ipython.org/gist/cparmer/7721116
        #x=df.index.values.astype('datetime64[s]')
    else:
        x = series.index.values

    line = {}
    line['x'] = x
    line['y'] = series.values
    line['name'] = name

    return [line]
  
  
  
  
  
  
  
