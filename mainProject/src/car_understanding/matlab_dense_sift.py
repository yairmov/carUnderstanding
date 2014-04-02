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


  cmd_params = "-nodisplay -nodesktop -nosplash -r tmp_dir_name='{}'; dense_sift; quit;".format(directory_name)

  matlab(cmd_params)
