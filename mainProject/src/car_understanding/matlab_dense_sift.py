'''
Functions to calculate dense sift by calling the matlab wrapper for VLFEAT.
I could not get the python wrapper to VLFEAT to work.
Created on Apr 1, 2014

@author: ymovshov
'''

# requires that the matlab executable is in the path.
from sh import matlab
import numpy as np
import scipy.io as sio
import os
from path import path
from sklearn.externals.joblib import Parallel, delayed, dump
import shutil

from dense_SIFT import normalize_sift
import util

def dense_sift_matlab(data_annos, config):
  print('Calling MATLAB to calculate SIFT')
  p = path(config.dataset.main_path)
  img_names = np.array(data_annos.img_path.apply(lambda x : str(x)))
  data_names = data_annos.basename.map(lambda x: str(os.path.splitext(x)[0]))
  p = path(config.SIFT.matlab.raw_dir)
  data_names = data_names.map(lambda x: str(p.joinpath(x + '.mat')))
  
  run_dense_sift_matlab(img_names, data_names, config.SIFT.matlab.sizes)
  
  print('Normalizing SIFT descriptors')
  normalize_sift_data(data_annos, config)


def run_dense_sift_matlab(img_names, data_names, sizes):
  '''
  Calculates dense sift using matlab.
  img_names - list of paths to images.
  data_names - list of paths of where to save the results. result of img_names[i]
  will be saved in a flie called data_names[i].
  '''
  # convert lists to numpy object arrays. These will be loaded as cell arrays
  # in matlab.

  img_cell = np.array(img_names, dtype=np.object)
  data_cell = np.array(data_names, dtype=np.object)
  directory_name = './tmp'
  util.makedir_if_needed(directory_name)
  
  sio.savemat(os.path.join(directory_name, 'data.mat'),
               {'img_cell':img_cell, 'data_cell': data_cell,
                'sizes': sizes})


  cmd_params = '''-nodisplay -nodesktop -nosplash -r "dense_sift('{}'); quit" '''.format(directory_name)


  print 'calling matlab with params: {}'.format(cmd_params)
  res = matlab(cmd_params)
  print res
  
  # remove the tmp dir
  shutil.rmtree(directory_name)


def normalize_one(name):
  out_name = path(name).splitext()[0] + '.dat'
  a = sio.loadmat(name)
  desc = a['desc']
  frames = a['frames']
  normalize_sift(desc, inplace=True)
  dump(dict(frames=frames, desc=desc), out_name, compress=3)

def normalize_sift_data(data_annos, config):

  data_names = data_annos.basename.map(lambda x: str(os.path.splitext(x)[0]))
  p = path(config.SIFT.matlab.raw_dir)
  data_names = np.array(data_names.map(lambda x: str(p.joinpath(x + '.mat'))))
  
  Parallel(n_jobs=config.n_cores, verbose=config.logging.verbose)(
                 delayed(normalize_one)(data_names[ii])
                 for ii in range(len(data_names)))

#   pbar = util.ProgressBar(len(data_names))
#   for ii, name in enumerate(data_names):
#     out_name = path(name).splitext()[0] + '.dat'
#     a = sio.loadmat(name)
#     desc = a['desc']
#     frames = a['frames']
#     normalize_sift(desc, inplace=True)
#     dump(dict(frames=frames, desc=desc), out_name, compress=3)
#     pbar.animate(ii)


