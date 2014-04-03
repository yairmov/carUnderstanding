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
import tempfile
import os
from path import path



def dense_sift_matlab(data_annos, config):
  p = path(config.dataset.main_path)
  img_names = data_annos.rel_path.map(lambda x: str(p.joinpath(x)))
  data_names = data_annos.basename.map(lambda x: str(os.path.splitext(x)[0]))
  p = path(config.SIFT.matlab.raw_dir)
  data_names = data_names.map(lambda x: str(p.joinpath(x)))

  run_dense_sift_matlab(img_names, data_names)


def run_dense_sift_matlab(img_names, data_names):
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
#   directory_name = tempfile.mkdtemp()
  directory_name = './tmp'

  sio.savemat(os.path.join(directory_name, 'data.mat'),
               {'img_cell':img_cell, 'data_cell': data_cell})


  cmd_params = '''-nodisplay -nodesktop -nosplash -r "dense_sift('{}'); quit" '''.format(directory_name)
  # cmd_params = '-nodisplay -nodesktop -nosplash -r "dense_sift; quit" '


  print 'calling matlab with params: {}'.format(cmd_params)
  matlab(cmd_params)
