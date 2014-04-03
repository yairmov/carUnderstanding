#!/usr0/home/ymovshov/Enthought/Canopy_64bit/User/bin/python
'''
Created on Mar 13, 2014

Util module to do training and testing of the attribte classifiers.
@author: ymovshov
'''

import sys
import os
import numpy as np
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
# from mpltools import style
# style.use('ggplot')


from configuration import get_config
import fgcomp_dataset_utils as fgu
from attribute_classifier import AttributeClassifier
from attribute_selector import AttributeSelector
from util import ProgressBar
import Bow

__date__ = '2014-03-13'


def main(argv=None):  # IGNORE:C0111
  '''Command line options.'''

  if argv is None:
      argv = sys.argv
  else:
      sys.argv.extend(argv)
      
      
  program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
  program_license = '''%s

  Created by yair on %s.
  Copyright 2014 Yair Movshovitz-Attias. All rights reserved.
  
  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.
  
  USAGE train_and_test_attribute_classifiers <options>
  ''' % (program_shortdesc, str(__date__))
  parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
  parser.add_argument("-v", "--verbose", dest="verbose", action="count", help="set verbosity level [default: %(default)s]")
  
  parser.add_argument(dest="attrib_names", help="attributes to train/test [default: %(default)s]", nargs='+', default=None)
#   parser.add_argument("-c", "--crossval", dest="cv", default=False, action='store_true')
  parser.add_argument("-g", "--grid-search", dest="grid_search", default=False, action='store_true')
  
  # train flag
  parser.add_argument('--train',dest='train',action='store_true')
  parser.add_argument('--no-train',dest='train',action='store_false')
  parser.set_defaults(train=True)
  # test flag
  parser.add_argument('--test',dest='test',action='store_true')
  parser.add_argument('--no-test',dest='test',action='store_false')
  parser.set_defaults(test=True)
  # plot flag
  parser.add_argument('--plot',dest='plot',action='store_true')
  parser.add_argument('--no-plot',dest='plot',action='store_false')
  parser.set_defaults(plot=True)
  # overwrite flag
  parser.add_argument('--overwrite',dest='overwrite',action='store_true')
  parser.add_argument('--no-overwrite',dest='overwrite',action='store_false')
  parser.set_defaults(overwrite=True)
  
  # Process arguments
  args = parser.parse_args()
  
  print("Got arguments: ")
  print(args)

  config = get_config(args.attrib_names)
  (dataset, config) = fgu.get_all_metadata(config)
  
  if args.train:
    train(args, config, dataset)
  
  if args.test:
    test(args, config, dataset)
  
def test(args, config, dataset):
  print("Testing")
  print("========")
  print("")
  test_annos = dataset['test_annos']
#   test_annos = dataset['train_annos']
  attrib_selector = AttributeSelector(config, dataset['class_meta'])
  
  print "Load image Bow histograms from disk"
  features = np.empty(shape=[len(test_annos), config.SIFT.BoW.num_clusters])
  progress = ProgressBar(len(test_annos))
  for ii in range(len(test_annos)):
    img_name = test_annos.iloc[ii]['basename']
    img_name = os.path.splitext(img_name)[0]
    hist_filename = os.path.join(config.SIFT.BoW.hist_dir,
                                 img_name) + '_hist.dat'
    hist = Bow.load(hist_filename) 
    features[ii, :] = hist
    progress.animate(ii)
  print("")
  
  print("Apply classifiers")
  res = {}
  for ii, attrib_name in enumerate(config.attribute.names):
    print(attrib_name)
    print("")
    attrib_clf = AttributeClassifier.load('../../../attribute_classifiers/{}.dat'.format(attrib_name))
    curr_res = attrib_clf.decision_function(features, 
                                            use_prob=config.attribute.use_prob)  
    res[attrib_clf.name] = curr_res.reshape(len(curr_res))
  
  res = pd.DataFrame(data=res, index=test_annos.index)
  res = pd.concat([res, test_annos.ix[:, ['class_index']]], axis=1)
  
  K = np.ceil(np.sqrt(len(args.attrib_names)))
  for ii, attrib_name in enumerate(args.attrib_names):
    pos_classes = attrib_selector.class_ids_for_attribute(attrib_name)
    true_labels = np.array(res.class_index.isin(pos_classes))
    print("--------------{}-------------".format(attrib_name)) 
    
    print(classification_report(true_labels, np.array(res[str.lower(attrib_name)]) > config.attribute.high_thresh, 
                                target_names=['not-{}'.format(attrib_name),
                                              attrib_name]))
    
    print("classifier-score stats:")
    print(res[str.lower(attrib_name)].describe())
    print("-----------------------------")
    
    
    precision, recall, thresholds = precision_recall_curve(true_labels, 
                                                           np.array(res[str.lower(attrib_name)]))
    score = auc(recall, precision)
    print("Area Under Curve: %0.2f" % score)
    print ("")
    if args.plot:
      # Create the plot
      plt.subplot(K,K,ii+1)
      plt.plot(recall, precision, label='Precision-Recall curve')
      plt.title('Precision-Recall: {}'.format(attrib_name))
      plt.xlabel('Recall')
      plt.ylabel('Precision')
      plt.legend(['area = {}'.format(score)])
    
  if args.plot:
    plt.draw()
    plt.show()
    
    
  
def train(args, config, dataset):
  print("Training")
  print("========")
  print("")
  train_annos = dataset['train_annos']
  attrib_selector = AttributeSelector(config, dataset['class_meta'])
  for attrib_name in config.attribute.names:
    print(attrib_name)
    print("")
    fname = os.path.join(config.attribute.dir, attrib_name + '.dat')
    if (not os.path.isfile(fname)) or args.overwrite:
      pos_class_ids = attrib_selector.class_ids_for_attribute(attrib_name)
      pos_img_ids = train_annos[train_annos.class_index.isin(pos_class_ids)].index
      attrib_clf = AttributeClassifier(config,
                                       train_annos,
                                       pos_img_ids,
                                       attrib_name,
                                       desc=attrib_name)
  
      attrib_clf.run_training_pipeline(grid_search=args.grid_search)
      AttributeClassifier.save(attrib_clf, fname)


    print( "-------------------------------------")
    print( "-------------------------------------")
    print("")

if __name__ == '__main__':
    main()
  
  
  
  
  
  