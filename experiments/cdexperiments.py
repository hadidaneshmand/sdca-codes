from __future__ import print_function

import numpy as np
import numpy.random
import numpy.linalg

#import sys
#import os
#lib_path = os.path.abspath('cppcd')
#sys.path.append(lib_path)
#import cppcd
#import lasso


def gen_normalized_gaussian_matrix(m, n):
  A = np.random.randn(m, n)
  # normalize
  col_norms = np.linalg.norm(A, axis=0)
  A = np.divide(A, col_norms)
  return A


def gen_sparse_rademacher_vector(n, k):
  x = np.zeros(n)
  coords = np.random.choice(n, k, replace=False)
  x[coords] = 1.0 - 2.0 * np.random.randint(0, 2, k)
  return x


def get_pyplot_data_iter(stats, opt=None):
  xvals = np.array([x for x, y in stats])
  yvals = np.array([y for x, y in stats])
  if not opt:
    opt = np.min(yvals)
  yvals -= opt
  yvals /= np.max(yvals)
  return xvals, yvals


def get_plot_data_time(progress, opt=None):
  xvals = [te.elapsed_time for te in progress]
  yvals = [te.current_objective_value for te in progress]
  if not opt:
    opt = np.min(yvals)
  yvals -= opt
  yvals /= np.max(yvals)
  return xvals, yvals

def get_plot_data_iter(progress, opt=None):
  yvals = [te.current_objective_value for te in progress]
  if not opt:
    opt = np.min(yvals)
  yvals -= opt
  yvals /= np.max(yvals)
  return yvals
