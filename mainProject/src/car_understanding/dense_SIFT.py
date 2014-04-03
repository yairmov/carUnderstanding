#!/usr/local/bin/python2.7
# encoding: utf-8
'''
car_understanding.dense_SIFT -- calculate dense SIFT on a list of images.

car_understanding.dense_SIFT is a tool for calculating dense SIFT on a list of images.

@author:     Yair Movshovitz-Attias

@copyright:  2014 Yair Movshovitz-Attias. All rights reserved.

@contact:    yair@cs.cmu.edu

@deffield    updated: Updated
'''

import sys
import os
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
import cv2 as cv
# import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.externals.joblib import load, dump

__all__ = []
__version__ = 0.1
__date__ = '2014-01-13'
__updated__ = '2014-01-13'

DEBUG = 0
TESTRUN = 0
PROFILE = 0

class CLIError(Exception):
  '''Generic exception to raise and log different fatal errors.'''
  def __init__(self, msg):
    super(CLIError).__init__(type(self))
    self.msg = "E: %s" % msg
  def __str__(self):
    return self.msg
  def __unicode__(self):
    return self.msg



'''
Most of the work is done here.
filename - Full path to image file or image as numpy array.
grid_spacing - spacing (in pixels) between locations on which SIFT is calculated (default: 4).
'''
def dense_SIFT(filename, grid_spacing=4):

  if isinstance(filename, basestring):
    img = cv.imread(filename)
  else:
    img = filename


  detector = cv.FeatureDetector_create("Dense")
  detector.setInt('initXyStep', grid_spacing)

  descriptor = cv.DescriptorExtractor_create("SIFT")

  kp = detector.detect(img)
  (kp, desc) = descriptor.compute(img, kp)

  # A bit hacky, but known to help classification accuracy later
  normalize_sift(desc, inplace=True)

  return (kp, desc)


def normalize_sift(sift_arr, inplace=True):
  '''
  Normalize sift descriptors.
  Descriptors who's norm is greater than one are normalized to unit,
  then large values (>0.2) are clipped, and the vectors are renormalized.
  '''
  # Find indices of descriptors to be normalized (those whose norm is larger than 1)
  normalize_ind = np.linalg.norm(sift_arr, 2, axis=1) > 1
  sift_arr_norm = sift_arr[normalize_ind, :]

  # Normalize them to 1
  sift_arr_norm = preprocessing.normalize(sift_arr_norm, norm='l2', axis=1)

  # Suppress large gradients
  sift_arr_norm[sift_arr_norm > 0.2] = 0.2

  # Finally, renormalize to unit length
  sift_arr_norm = preprocessing.normalize(sift_arr_norm, norm='l2', axis=1)

  # check if to make copy of the array
  if inplace:
    ret_arr = sift_arr
  else:
    ret_arr = sift_arr.copy()

  ret_arr[normalize_ind, :] = sift_arr_norm

  # Return pointer to modified array
  return ret_arr




def save_to_disk(kp, desc, out_filename):
  # create a dictionary with the saved data
  # You can't just pickle cv2.Keypoint (cause opencv sucks) so I need to
  # do some manual conversions here
  keypoints_as_dict = []
  for point in kp:
    tmp = dict(angle=point.angle,
               class_id=point.class_id,
               octave=point.octave,
               pt=point.pt,
               response=point.response,
               size=point.size)
    keypoints_as_dict.append(tmp)

  saved_data = dict(keypoints_as_dict=keypoints_as_dict, desc=desc)

  dump(saved_data, out_filename, compress=3)


def load_from_disk(infilename, matlab_version=False):

  saved_data = load(infilename)

  desc = saved_data['desc']

  if matlab_version:
    kp = saved_data['frames']
  else:
    # now to convert the data back to opencv's cv2.Keypoint
    kp = []
    keypoints_as_dict = saved_data['keypoints_as_dict']
    for point in keypoints_as_dict:
      tmp = cv.KeyPoint(x=point['pt'][0], y=point['pt'][1], _size=point['size'],
                         _angle=point['angle'], _response=point['response'],
                         _octave=point['octave'], _class_id=point['class_id'])

      kp.append(tmp)
  return (kp, desc)


def main(argv=None):  # IGNORE:C0111
  '''Command line options.'''

  if argv is None:
      argv = sys.argv
  else:
      sys.argv.extend(argv)

  program_name = os.path.basename(sys.argv[0])
  program_version = "v%s" % __version__
  program_build_date = str(__updated__)
  program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
  program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
  program_license = '''%s

Created by yair on %s.
Copyright 2014 Yair Movshovitz-Attias. All rights reserved.

Distributed on an "AS IS" basis without warranties
or conditions of any kind, either express or implied.

USAGE dense_SIFT <input image names>
''' % (program_shortdesc, str(__date__))

  try:
    # Setup argument parser
    parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", dest="verbose", action="count", help="set verbosity level [default: %(default)s]")
    parser.add_argument('-V', '--version', action='version', version=program_version_message)
    parser.add_argument('-O', '--out', type=str, help="path to folder in which binary files will be saved.", default='.')
    parser.add_argument(dest="paths", help="paths to input image file(s) [default: %(default)s]", metavar="file", nargs='+')

    # Process arguments
    args = parser.parse_args()

    paths = args.paths
    verbose = args.verbose
    out_path = args.out

    # If out_path does not exist, create it (not thread safe)
    if not os.path.isdir(out_path):
      os.makedirs(out_path)

    if verbose > 0:
      print("Verbose mode on")


    for inpath in paths:
      print 'Extracting dense-SIFT from image:', inpath, '...',
      (kp, desc) = dense_SIFT(inpath)
      print 'Done.'
      # Replace extension to .dat and place in out_path
      (name, ext) = os.path.splitext(os.path.split(inpath)[1])
      save_name = os.path.join(out_path, name + '.dat')
      print 'Saving dense-SIFT to file: "', save_name, '" ...',
      save_to_disk(kp, desc, save_name)
      print 'Done.'
    return 0

  except Exception, e:
    if DEBUG or TESTRUN:
      raise(e)
    indent = len(program_name) * " "
    sys.stderr.write(program_name + ": " + repr(e) + "\n")
    sys.stderr.write(indent + "  for help use --help")
    return 2

if __name__ == "__main__":
  if DEBUG:
    sys.argv.append("-h")
    sys.argv.append("-v")
  if TESTRUN:
    import doctest
    doctest.testmod()
  if PROFILE:
    import cProfile
    import pstats
    profile_filename = 'car_understanding.dense_SIFT_profile.txt'
    cProfile.run('main()', profile_filename)
    statsfile = open("profile_stats.txt", "wb")
    p = pstats.Stats(profile_filename, stream=statsfile)
    stats = p.strip_dirs().sort_stats('cumulative')
    stats.print_stats()
    statsfile.close()
    sys.exit(0)
  sys.exit(main())
