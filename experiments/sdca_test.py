from __future__ import print_function
from sdca import *
from sklearn import svm
from sklearn import preprocessing

import sys
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1","True")

print('Number of arguments:{}'.format( len(sys.argv)))
print('Argument List:{}'.format(str(sys.argv)))

n = int(sys.argv[1]); 
d = int(sys.argv[2]); 
filename = sys.argv[3]
y, X = readfile(filename,n,d)

print('Reading data accomblished')

       
if(len(sys.argv) >10):
  sub_n = int(sys.argv[9])
  inds = np.random.permutation(n)
  X_hat = np.zeros((sub_n,d))
  y_hat = np.zeros(sub_n)
  for i in range(sub_n): 
      y_hat[i] = y[inds[i]]
      X_hat[i,:] = X[inds[i],:]
  y = y_hat
  X = X_hat

for i in range(len(sys.argv)): 
    if(sys.argv[i] == "scale"): 
        print("scaling data")
        X = preprocessing.scale(X)
    if(sys.argv[i] == "normalize"):
        print("normalizing data")
        X = preprocessing.normalize(X)
    if(sys.argv[i] == "binarize"):
        print("binarizing data")
        X = np.fabs(X)
        binarizer = preprocessing.Binarizer().fit(X)
        X = binarizer.transform(X)
        
A = np.zeros(X.shape)
n_a, d_a = A.shape
for i in range(n_a): 
    if(y[i]==2 or y[i] == 0):
        y[i] = -1;
    A[i,:] = y[i]*X[i,:]

lambd = float(sys.argv[4])
passes = int(sys.argv[5])
f_out = sys.argv[6]
server_i = str2bool(sys.argv[7])
run_experiment(A,y,X,lambd,passes,f_out,server = server_i,st = int(sys.argv[8]))

