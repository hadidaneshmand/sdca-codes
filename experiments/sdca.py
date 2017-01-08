from __future__ import print_function
import math
from sklearn import svm
import numpy as np
import time
import sys
import os
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

def indices_get(a, b_list):
    return [val for (i, val) in enumerate(a) if (b_list[val] == 0)]

def angle(points, query):
   pn = preprocessing.normalize(points)
   qn = np.divide(query,np.linalg.norm(query))
   return np.arccos(np.transpose(np.dot(pn,np.transpose(qn))))

def get_data_plot(stats):
    xvals = np.array([x1 for x1, y1 in stats])
    yvals = np.array([y1 for x1, y1 in stats])
    return (xvals,yvals)

def readfile(filename, n,d):
    y = np.zeros(n) # targets
    X = np.zeros((n,d)) # input matrix each row is a sample data point
    li = 0 
    with open(filename, "rb") as f:
        for line in f:
           if li>=n : 
             break;
           parts = line.split()
           y[li] = float(parts[0])
           for i in range(len(parts)): 
                if i >0 and parts[i] != '\n': 
                    fparts = parts[i].split(":")
                    X[li,int(fparts[0])-1] = float(fparts[1])
           li = li +1
    return (y,X)

def w_alpha(alpha,X,lambd,verbose=False): # check the equation 3 of (SDCA) paper
    n, d = X.shape
    w = np.zeros(d)
    for i in range(n): 
        w = w + alpha[i]*X[i,:]/lambd
    w = w/n
    return w


def dual_obj(w, alpha,X,lambd,verbose=False): # computes the dual function value for a given parameter alpha 
    da = 0
    n, d = X.shape
    for i in range(n): 
        if alpha[i] <= 1 and alpha[i] >= 0:
            da = da + alpha[i]
        else:
            da = da + float("inf")
    da = da / n 
    da = da - 0.5*lambd*np.square(np.linalg.norm(w_alpha(alpha,X,lambd)))
    return da

def primal_func(w,X,lambd,verbose = False): # computes the primal value for the given parameter w
    n, d = X.shape
    pw = 0
    for i in range(n): 
        pw = pw + max(0,1-np.dot(w,X[i,:]))
    pw = pw/n 
    pw = pw + 0.5*lambd*np.square(np.linalg.norm(w))
    return pw
def meta_svm(A, lambd, num_effective_passes, order_generator, verbose=False, obj = dual_obj, queue_size = -1,active_strategy = -1):
  #each row contains a data point with dim d
  # active_stargey: case -1 (nothing), case 1 (reset the block list in each iteration), case 2 ( holding a queue with a limited size), case 3 (online random permutation replacement) 
  inv_sq_row_norms = np.divide(1.0, np.square(np.linalg.norm(A, axis=1)))
  n, d = A.shape
  rperm = np.random.permutation(n);
  permi = 0
  if queue_size == -1: 
     queue_size = int(n/3)
  alpha = np.zeros(n)
  w = w_alpha(alpha,A,lambd)
  obj_value = obj(w,alpha,A,lambd)
  q = Queue()
  if verbose:
    print(" Initial objective value: {}".format(obj_value))
  past_ef_pass = 0
  b_list = np.zeros(n);
  stats = [(0.0, obj_value)]
  completed_effective_passes = 0.0
  past_w = w
  past_obj = obj_value
  while completed_effective_passes < num_effective_passes:
    coords = order_generator( A, w,b_list)
    blocked_c = 0; 
    delta_sum = 0; 
    co = 0
    if active_strategy == 1: 
     if (int(past_ef_pass) != int(completed_effective_passes)) and (int(completed_effective_passes)%2 ==0): 
       b_list = np.zeros(n)
     if len(coords) == 0: 
       b_list = np.zeros(n)
       continue
    
    for ii in coords:
        co  = co + 1
        alpha_i_old = alpha[ii]
        xi = A[ii,:]
        inv_sqi = inv_sq_row_norms[ii]
       
        # coordinate update step 
        delta_alpha_i = max(0,min(1,(lambd*n*(1-np.dot(xi,w))*inv_sqi)+alpha_i_old)) - alpha_i_old
        if co <10: 
            if verbose:
              print("delta_alpha[{}]= {},dot = {},alpha_old ={}".format(ii,delta_alpha_i,1-np.dot(xi,w),alpha_i_old))
        delta_sum = delta_sum + abs(delta_alpha_i)
        if delta_alpha_i == 0 or abs(delta_alpha_i)< 0.000001:
           blocked_c = blocked_c + 1
           if active_strategy == 1 or active_strategy == 2: 
             b_list[ii] = 1
           if active_strategy == 2:
              q.enqueue(ii)
              if q.size()> queue_size:
                 ij = q.dequeue()
                 b_list[ij] = 0 
           if active_strategy == 3: 
             ii = rperm[permi]
             
             permi = permi + 1 
             if permi == n: 
                rperm = np.random.permutation(n);
                permi = 0
             alpha_i_old = alpha[ii]
             xi = A[ii,:]
             inv_sqi = inv_sq_row_norms[ii]
             # coordinate update step 
             delta_alpha_i = max(0,min(1,(lambd*n*(1-np.dot(xi,w))*inv_sqi)+alpha_i_old)) - alpha_i_old
        # update the dual and primal paramters
        alpha[ii] = alpha[ii] + delta_alpha_i
        w = w + delta_alpha_i*xi/(lambd*n)
    
    past_ef_pass = completed_effective_passes
    completed_effective_passes += len(coords) / float(n)
    obj_value = obj(w,alpha,A,lambd)
    stats.append((completed_effective_passes, obj_value))
    if verbose:
      print("Obj[{}]= {}, ratio of blocked = {}, delta obj={}".format(completed_effective_passes, obj_value, float(blocked_c)/len(coords),  (obj_value - past_obj)))
    past_w = w 
    past_obj = obj_value
  return (alpha, stats,w)

def svm_randomiid( A, lambd, num_passes, verbose=False,obj = dual_obj,active_s = -1 ):
  n, d = A.shape
  def rand_perm_ex_order_generator(A, r,b_list):
    inds = np.random.randint(n, size=n)
    return indices_get(inds, b_list)
  if verbose:
    print("#########################################")
    print("Random perm exclusive coordinate descent svm solver")
  return meta_svm( A, lambd, num_passes, rand_perm_ex_order_generator, verbose,obj,active_strategy = active_s)

def svm_cyclic( A, lambd, num_passes, verbose=False,obj = dual_obj):
  n,d = A.shape
  def cyclic_order_generator(*unused):
    return range(n)
  if verbose:
    print("#########################################")
    print("Cyclic coordinate descent svm solver")
  return meta_svm( A, lambd, num_passes, cyclic_order_generator, verbose,obj)
def svm_randomperm( A, lambd, num_passes, verbose=False,obj = dual_obj,active_s = -1):
  n, d = A.shape
  def randomperm_order_generator(A, r,b_list):
    inds = np.random.permutation(n)
    return indices_get(inds, b_list)
  if verbose:
    print("#########################################")
    print("Random permutation coordinate descent svm solver")
  return meta_svm( A, lambd, num_passes, randomperm_order_generator, verbose,obj, active_strategy = active_s)



def svm_maxip_test( A, lambd, num_passes, prefix_length=-1, verbose=False, obj = dual_obj, martin = False, active_s = -1):
  n , d = A.shape
  X = np.zeros((n,d+1))
  for i in range(n): 
    X[i,0:d] = A[i,:]
    X[i,d]=1
  AT = X.astype(np.float32)
  if prefix_length < 0:
    prefix_length = n
  def maxip_order_generator(y,r,b_list): 
    ips = 1-np.dot(A, r)
    res1 = np.flipud(np.argsort(ips))
    res2 = np.flipud(np.argsort(-1*ips))
    seq = [index for pair in zip(res1, res2) for index in pair]
    if verbose: 
        print("indices_all:{}".format(seq[:10]))
        print("sum b:{}".format(sum(b_list)))
    seq = indices_get(seq, b_list)
    if verbose:
        print("indices_unblocked:{}".format(seq[:10]))
        print("--maxip dot:{}".format(np.abs(ips[seq[:10]])))
#         print("--maxip angles:{}".format(angle(AT[seq[:10],:],r2)))
        print("---------")
    return seq[:prefix_length]
  if verbose:
    print("#########################################")
    print("Maximum inner product coordinate descent svm solver")
  return meta_svm( A, lambd, num_passes, maxip_order_generator, verbose,obj, active_strategy = active_s)

def svm_maxip( A, lambd, num_passes, prefix_length=-1, verbose=False, obj = dual_obj, martin = False, active_s = -1):
  n , d = A.shape
  X = np.zeros((n,d+1))
  for i in range(n): 
    X[i,0:d] = A[i,:]
    X[i,d]=1
  AT = X.astype(np.float32)
  if prefix_length < 0:
    prefix_length = n
  def maxip_order_generator(y,r,b_list):
    r2 = np.zeros(d+1);
    r2[0:d] = r; 
    r2[d] = -1; 
    ips = 1-np.dot(A, r)
    indices = np.flipud(np.argsort(np.abs(ips)))
    if verbose: 
        print("indices_all:{}".format(indices[:10]))
        print("sum b:{}".format(sum(b_list)))
    indices = indices_get(indices, b_list)
    if verbose:
        print("indices_unblocked:{}".format(indices[:10]))
        print("--maxip dot:{}".format(np.abs(ips[indices[:10]])))
        print("--maxip angles:{}".format(angle(AT[indices[:10],:],r2)))
        print("---------")
    return indices[:prefix_length]
  if verbose:
    print("#########################################")
    print("Maximum inner product coordinate descent svm solver")
  return meta_svm( A, lambd, num_passes, maxip_order_generator, verbose,obj, active_strategy = active_s)

def svm_GL( A, lambd, num_passes, prefix_length=-1, verbose=False, obj = dual_obj, martin = False, active_s = -1):
  n , d = A.shape
  X = np.zeros((n,d+1))
  inv_sq_row_norms = np.divide(1.0, np.square(np.linalg.norm(A, axis=1)))
  if prefix_length < 0:
    prefix_length = n
  def gl_order_generator(y,r,b_list):
    ips = (1-np.dot(A, r))*inv_sq_row_norms
    indices = np.flipud(np.argsort(np.abs(ips)))
    indices = indices_get(indices, b_list)
    if verbose:
        print("indices_unblocked:{}".format(indices[:10]))
        print("-- GL value:{}".format(np.abs(ips[indices[:10]])))
        print("---------")
    return indices[:prefix_length]
  if verbose:
    print("#########################################")
    print("GL coordinate descent svm solver")
  return meta_svm( A, lambd, num_passes, gl_order_generator, verbose,obj, active_strategy = active_s)

