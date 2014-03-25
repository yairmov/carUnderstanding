'''
Util functions for the carUnderstanding projects
Created on Jan 28, 2014

@author: yair@cs.cmu.edu
'''

from __future__ import print_function
import os
import scipy.misc
import matplotlib.pyplot as plt
from clint.textui import progress
import sys
import cv2 as cv
import base64
import numpy as np

def set_width_to_normalize_bb(img, xmin, xmax, to_width):
  w = xmax - xmin
  s = to_width / w
  img = scipy.misc.imresize(img, s)
  return img, s

def change_bb_loc(scaler, xmin, xmax, ymin, ymax):
  w, h = (xmax - xmin, ymax - ymin)
  nw, nh = (scaler * w, scaler * h)

  center = (xmin + (w/2), ymin + (h/2))
  new_center = (center[0] * scaler, center[1] * scaler)

  xmin, xmax = (new_center[0] - (nw/2), new_center[0] + (nw/2))
  ymin, ymax = (new_center[1] - (nh/2), new_center[1] + (nh/2))


  return xmin, xmax, ymin, ymax

'''
This will change the images in the dataset!!
Call this on a COPY of the dataset
'''
def normalize_dataset(train_annos_file, main_path, out_file, bb_width):
  # read lines from file
  with open(os.path.join(main_path, train_annos_file)) as f:
    content = f.readlines()

  out_fid = open(os.path.join(main_path, out_file), 'w')
  print("Resizing images such that BB is of width = %g" % bb_width)
  for ii in progress.bar(range(len(content))):
    curr_line = content[ii]
    curr_line = curr_line.strip()
    (img_index, rel_path, domain_index,
     class_index, xmin, xmax, ymin, ymax) = curr_line.split(',')

    xmin, xmax, ymin, ymax = (float(x) for x in (xmin, xmax, ymin, ymax))

    # Read image and resize such that bounding box is of specific size
    img_file = os.path.join(main_path, rel_path)
    img = scipy.misc.imread(img_file)
    img, scaler = set_width_to_normalize_bb(img, xmin, xmax, bb_width)

    # Write image back to disk
    scipy.misc.imsave(img_file, img)

    # change xmin, xmax, ymin, ymax to the new size
    xmin, xmax, ymin, ymax = change_bb_loc(scaler, xmin, xmax, ymin, ymax)
    xmin, xmax, ymin, ymax = (str(int(x)) for x in (xmin, xmax, ymin, ymax))

    # generate new text line
    new_line = ','.join((img_index, rel_path, domain_index,
              class_index, xmin, xmax, ymin, ymax))

    out_fid.write("%s\n" % new_line)

  out_fid.close()


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
  for ii in range(0, len(annos)):
    curr_anno = annos.iloc[ii]
    img = scipy.misc.imread(os.path.join(
                      config.dataset.main_path, curr_anno['rel_path']))
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


class ProgressBar:
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 50
        self.__update_amount(0)

    def animate(self, iter):
        print('\r', self, end='')
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)



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


def create_image_page(img_files, html_file, width=200, num_per_row=9,
                                  split_every=np.Inf, usr_str=''):

  k = 0;

  width_str = 'width = ' + str(width)
  html_code_image = '<img src="data:image/jpeg;base64, {}" ' + width_str + ' style="border:1px solid white" >'

  html_str = '<html><body> '

  # Create HTML string
  for f in img_files:
    with open(f, "rb") as img_file:
      img_str_64 = base64.b64encode(img_file.read())

#     img_name = f
    s = html_code_image.format(img_str_64)
    html_str += s + '\n'
    k += 1

    if (k % split_every == 0):
      html_str += '<hr>' + '<center> <h2> ' + usr_str + '</h2></center> <br> <hr>'

    if (k == num_per_row):
      k = 0
      html_str += '<hr>'

  html_str += '</body></html>'

  # Write to file
  with open(html_file, "w") as out_file:
    out_file.write(html_str)




def plot_dataset_embedding(dataset, config, 
                           labels=None, title=None,
                           show_images=True):
  '''
  Display a figure that shows an 2D embedding of the BoW features for the
  dataset. It also loads the images of the data, and displays them on the 
  figure (when there is enough space)
  '''
  import Bow
  import pandas as pd
  import numpy as np
  from sklearn import ensemble, decomposition, manifold
  from PIL import Image
  from path import path
  
  n_items = dataset.shape[0]
  features = np.empty(shape=[n_items, config.SIFT.BoW.num_clusters])
  p_bar = ProgressBar(n_items)
  print('Loading {} BoW from disk'.format(n_items))
  for ii in range(n_items):
    p_bar.animate(ii)
    img_name = dataset.iloc[ii]['basename']
    img_name = os.path.splitext(img_name)[0]
    hist_filename = os.path.join(config.SIFT.BoW.hist_dir,
                                 img_name) + '_hist.dat'
    hist = Bow.load(hist_filename) 
    features[ii, :] = hist
     
    
  if labels is None:
    labels = dataset.class_index
  
  
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
    img_names = dataset.rel_path.map(lambda x: p.joinpath(x))
    images = []
    for ii in range(len(dataset)):
      img_name = img_names.iloc[ii]
      im = Image.open(img_name)
      im.thumbnail([40,40])
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

  pl.figure(figsize=[10,10])
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
  

if __name__ == '__main__':
  pass


