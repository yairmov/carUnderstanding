'''
Created on Mar 13, 2014

Util module to do training and testing of the attribte classifiers.
@author: ymovshov
'''

import sys
import os
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

from configuration import get_config
import fgcomp_dataset_utils as fgu
from attribute_classifier import AttributeClassifier
from attribute_selector import AttributeSelector

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
  parser.add_argument("-c", "--crossval", dest="cv", default=False, action='store_true')
  
  # Process arguments
  args = parser.parse_args()

  config = get_config(args.attrib_names)
  (dataset, config) = fgu.get_all_metadata(config)
  
  
  train(args, config, dataset)
  
#   test(args, config, dataset)
  
def train(args, config, dataset):
  print("Training")
  train_annos = dataset['train_annos']
  attrib_selector = AttributeSelector(config, dataset['class_meta'])
  for attrib_name in args.attrib_names:
    print attrib_name
    pos_class_ids = attrib_selector.class_ids_for_attribute(attrib_name)
    pos_img_ids = train_annos[train_annos.class_index.isin(pos_class_ids)].index
    attrib_clf = AttributeClassifier(config,
                                     train_annos,
                                     pos_img_ids,
                                     attrib_name,
                                     desc=attrib_name)

    attrib_clf.run_training_pipeline(args.cv)
    
    AttributeClassifier.save(attrib_clf, os.path.join(config.attribute.dir,
                                                      attrib_clf.name + '.dat'))


    print "-------------------------------------"
    print "-------------------------------------"

if __name__ == '__main__':
    pass