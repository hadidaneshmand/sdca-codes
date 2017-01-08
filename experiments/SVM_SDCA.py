from __future__ import print_function

import math
import numpy as np
# this implementation is based on (SDCA) paper "Stochastic Dual Coordinate Ascent Methods for Regularised Loss
# Minimization" available on http://www.jmlr.org/papers/volume14/shalev-shwartz13a/shalev-shwartz13a.pdf

def w_alpha(alpha,X,lambd,verbose=False): # check the equation 3 of (SDCA) paper
    n, d = X.shape
    w = np.zeros(d)
    for i in range(n): 
        w = w + alpha[i]*X[i,:]/lambd
    w = w/n
    return w

def primal_func(w,y,X,lambd,verbose=False): # computes the primal value for the given parameter w
    n, d = X.shape
    pw = 0
    for i in range(n): 
        if verbose:
            print("P({}):y={},dot={},delta_pw={}".format(i,y[i],np.dot(w,X[i,:]),max(0,1-y[i]*np.dot(w,X[i,:]))))
        pw = pw + max(0,1-y[i]*np.dot(w,X[i,:]))
    pw = pw/n 
    pw = pw + 0.5*lambd*np.square(np.linalg.norm(w))
    return pw

def dual_func(alpha,y,X,lambd,verbose=False): # computes the dual function value for a given parameter alpha 
    da = 0
    n, d = X.shape
    for i in range(n): 
        if alpha[i]*y[i] <= 1 and alpha[i]*y[i] >= 0:
            da = da + alpha[i]*y[i]
        else:
            da = da + float("inf")
    da = da / n 
    da = da - 0.5*lambd*np.square(np.linalg.norm(w_alpha(alpha,X,lambd)))
    return da 



def SVM_SDCA(y, X, lambd, num_steps, T0, verbose=False):
  # y_n \in {-1,1} is target,
  # X_{n \time d}: is the row-wise arrange of observations , so each observation has dimension d 
  # Please note that this implementation just includes the output averaging option with paramter T0
  if verbose:
    print("Stochastic coordinate Ascent for SVM")
  inv_sq_row_norms = np.divide(1.0, np.square(np.linalg.norm(X, axis=1)))
  n, d = X.shape
  # it might be better to change initial value of alpha 
  alpha = np.zeros(n)
  #setting w(alpha_0)
  w = w_alpha(alpha,X,lambd)
  duals = np.zeros(num_steps)# dual function values per iteration
  primals = np.zeros(num_steps) #primal function values per iteration 
  if verbose:
    primal_value = primal_func(w,y,X,lambd)
    dual_value = dual_func(alpha,y,X,lambd)
    print("  Initial primal objective value: {}, Intial dual objective value: {}".format(primal_value, dual_value))
  w_bar = np.zeros(d)
  T_T0 = 0
  alpha_bar = np.zeros(n)
  for cur_step in range(num_steps):
    # pick a random dual coordinate
    ii = np.random.randint(0, n)
    alpha_i_old = alpha[ii]
    xi = X[ii,:]
    yi = y[ii]
    inv_sqi = inv_sq_row_norms[ii]
    # coordinate update step 
    delta_alpha_i = yi*max(0,min(1,(lambd*n*(1-yi*np.dot(xi,w))*inv_sqi)+alpha_i_old*yi)) - alpha_i_old
    # update the dual and primal paramters
    alpha[ii] = alpha[ii] + delta_alpha_i
    w = w + delta_alpha_i*xi/(lambd*n)
    
    # output averaging 
    if cur_step > T0:
       w_bar = w_bar + w
       alpha_bar = alpha_bar + alpha
       T_T0 = T_T0 +1 
    
    duals[cur_step] = dual_func(alpha,y,X,lambd)
    primals[cur_step] = primal_func(w,y,X,lambd)
    if verbose: #and cur_step % n == 0 and cur_step > 0
      print("step:{},primal: {},dual: {}\n".format(cur_step,primals[cur_step], duals[cur_step] ))
  return (w_bar/T_T0, alpha_bar/T_T0 ,primals,duals)


def SVM_SteepestDCA(y, X, lambd, num_steps, T0, verbose=False):
  # Computes change of all coordinates (delta_alphas) and updates the coordinate which have the maximum absolut change
  # y_n \in {-1,1} is target,
  # X_{n \time d}: is the row-wise arrange of observations , so each observation has dimension d 
  # Please note that this implementation just includes the output averaging option with paramter T0
  if verbose:
    print("Steepest Coordinate Ascent for SVM")
  inv_sq_row_norms = np.divide(1.0, np.square(np.linalg.norm(X, axis=1)))
  n, d = X.shape
  # it might be better to change initial value of alpha 
  alpha = np.zeros(n)
  #setting w(alpha_0)
  w = w_alpha(alpha,X,lambd)
  duals = np.zeros(num_steps)# dual function values per iteration
  primals = np.zeros(num_steps) #primal function values per iteration 
  if verbose:
    primal_value = primal_func(w,y,X,lambd)
    dual_value = dual_func(alpha,y,X,lambd)
    print("Initial primal objective value: {}, Intial dual objective value: {}".format(primal_value, dual_value))
  w_bar = np.zeros(d)
  T_T0 = 0
  alpha_bar = np.zeros(n)
  for cur_step in range(num_steps): 
    # Compute all the alphas 
    delta_alphas = np.multiply(y,np.maximum(0,np.minimum(1,(lambd*n*np.multiply((1-np.multiply(y,np.dot(X,w))),inv_sq_row_norms))+np.multiply(alpha,y))))-alpha
    # Pick the maximum one 
    ii = np.argmax(abs(delta_alphas))
    # update the dual and primal paramters
    xi = X[ii,:]
    alpha[ii] = alpha[ii] + delta_alphas[ii]
    w = w + delta_alphas[ii]*xi/(lambd*n)
    
    # output averaging 
    if cur_step > T0:
       w_bar = w_bar + w
       alpha_bar = alpha_bar + alpha
       T_T0 = T_T0 +1 
    
    duals[cur_step] = dual_func(alpha,y,X,lambd)
    primals[cur_step] = primal_func(w,y,X,lambd)
    if verbose: #and cur_step % n == 0 and cur_step > 0
      print("step:{},primal: {},dual: {}\n".format(cur_step,primals[cur_step], duals[cur_step] ))
  return (w_bar/T_T0, alpha_bar/T_T0 ,primals,duals)

def sortbyindeces(a):
  return [i[0] for i in sorted(enumerate(a), key=lambda a:a[1],reverse=True)]

def SVM_SteepestDualGradient_DCA(y, X, lambd, num_steps, T0, verbose=False): 
  # Computes the gradient of dual function and updates the coordinate with maximum absolute gradient value 
  # y_n \in {-1,1} is target,
  # X_{n \time d}: is the row-wise arrange of observations , so each observation has dimension d 
  # Please note that this implementation just includes the output averaging option with paramter T0
  if verbose:
    print("Steepest Dual Gradient Coordinate Ascent for SVM")
  inv_sq_row_norms = np.divide(1.0, np.square(np.linalg.norm(X, axis=1)))
  n, d = X.shape
  # it might be better to change initial value of alpha 
  alpha = np.zeros(n)
  #setting w(alpha_0)
  w = w_alpha(alpha,X,lambd)
  duals = np.zeros(num_steps)# dual function values per iteration
  primals = np.zeros(num_steps) #primal function values per iteration 
  if verbose:
    primal_value = primal_func(w,y,X,lambd)
    dual_value = dual_func(alpha,y,X,lambd)
    print("Initial primal objective value: {}, Intial dual objective value: {}".format(primal_value, dual_value))
  w_bar = np.zeros(d)
  T_T0 = 0
  alpha_bar = np.zeros(n)
  for cur_step in range(num_steps): 
    # Compute all delta alphas 
    #delta_alphas = np.multiply(y,np.maximum(0,np.minimum(1,(lambd*n*np.multiply((1-np.multiply(y,np.dot(X,w))),inv_sq_row_norms))+np.multiply(alpha,y))))-alpha
    
    # Compute the gradient 
    prods = np.dot(X,w)
    g_alpha = y - prods
    avg_pr = np.average(prods)
    max_pr = max(prods)
    cosines = np.dot(X,w)/(np.linalg.norm(X, axis=1)*np.linalg.norm(w))
    avg_cos = np.average(cosines)
    max_cos = max(cosines)
    
    # Sort indeces in descent order of the gradient
    inds = sortbyindeces(abs(g_alpha))
    ii = inds[1]
    # Choose the maximum coordinate which doesn't vailate the constratins for the update step 
    for i in range(len(g_alpha)): 
        j = inds[i]
        if alpha[j]!= 0 or ((alpha[j] == 0) and (g_alpha[j]*y[j] >= 0)):
            ii = j 
            break
                  
    # update the coordinate exactly same as SDCA
    alpha_i_old = alpha[ii]
    xi = X[ii,:]
    yi = y[ii]
    inv_sqi = inv_sq_row_norms[ii]
    # coordinate update step 
    delta_alpha_i = yi*max(0,min(1,(lambd*n*(1-yi*np.dot(xi,w))*inv_sqi)+alpha_i_old*yi)) - alpha_i_old
    # update the dual and primal paramters
    alpha[ii] = alpha[ii] + delta_alpha_i
    w = w + delta_alpha_i*xi/(lambd*n)
    
    # output averaging 
    if cur_step > T0:
       w_bar = w_bar + w
       alpha_bar = alpha_bar + alpha
       T_T0 = T_T0 +1 
    
    duals[cur_step] = dual_func(alpha,y,X,lambd)
    primals[cur_step] = primal_func(w,y,X,lambd)
    if verbose: #and cur_step % n == 0 and cur_step > 0
      print("step:{},primal: {},dual: {}, avg product = {}, max product = {}, avg cosine = {}, max cosine = {} \n".format(cur_step,primals[cur_step], duals[cur_step],avg_pr,max_pr,avg_cos,max_cos ))
  return (w_bar/T_T0, alpha_bar/T_T0 ,primals,duals)

def SVM_GSL_DCA(y, X, lambd, num_steps, T0, verbose=False): 
  # Computes the gradient of the dual function and updates the coordinate with maximum absolute gradient value over the lipschitz 
  # coordinate constant (for more information please check the section 6.2 of the paper http://jmlr.org/proceedings/papers/v37/nutini15.pdf 
  # y_n \in {-1,1} is target,
  # X_{n \time d}: is the row-wise arrange of observations , so each observation has dimension d 
  # Please note that this implementation just includes the output averaging option with paramter T0
  if verbose:
    print("GSL_DCA for SVM")
  inv_sq_row_norms = np.divide(1.0, np.square(np.linalg.norm(X, axis=1)))
  n, d = X.shape
  # it might be better to change initial value of alpha 
  alpha = np.zeros(n)
  #setting w(alpha_0)
  w = w_alpha(alpha,X,lambd)
  duals = np.zeros(num_steps)# dual function values per iteration
  primals = np.zeros(num_steps) #primal function values per iteration 
  innerproduct = np.zeros((num_steps,2)) # the first column contains the average inner products and the last on is maximum inner product
  if verbose:
    primal_value = primal_func(w,y,X,lambd)
    dual_value = dual_func(alpha,y,X,lambd)
    print("Initial primal objective value: {}, Intial dual objective value: {}".format(primal_value, dual_value))
  w_bar = np.zeros(d)
  T_T0 = 0
  alpha_bar = np.zeros(n)
  for cur_step in range(num_steps): 
    # Compute all delta alphas 
    #delta_alphas = np.multiply(y,np.maximum(0,np.minimum(1,(lambd*n*np.multiply((1-np.multiply(y,np.dot(X,w))),inv_sq_row_norms))+np.multiply(alpha,y))))-alpha
    
    # Compute the gradient 
    g_alpha = y - np.dot(X,w)
    
    # lipschitz involvment
    lg_alpha = abs(g_alpha)*np.sqrt(inv_sq_row_norms) 
    
    # Sort indeces in descent order of the gradient
    inds = sortbyindeces(lg_alpha)
    ii = inds[0]
    # Choose the maximum coordinate which doesn't violate the constratins for the update step 
    for i in range(len(g_alpha)): 
        j = inds[i]
        if alpha[j]!= 0 or ((alpha[j] == 0) and (g_alpha[j]*y[j] >= 0)):
            ii = j 
            break
    
    # update the coordinate exactly same as SDCA
    alpha_i_old = alpha[ii]
    xi = X[ii,:]
    yi = y[ii]
    inv_sqi = inv_sq_row_norms[ii]
    # coordinate update step 
    delta_alpha_i = yi*max(0,min(1,(lambd*n*(1-yi*np.dot(xi,w))*inv_sqi)+alpha_i_old*yi)) - alpha_i_old
    # update the dual and primal paramters
    alpha[ii] = alpha[ii] + delta_alpha_i
    w = w + delta_alpha_i*xi/(lambd*n)
    
    # output averaging 
    if cur_step > T0:
       w_bar = w_bar + w
       alpha_bar = alpha_bar + alpha
       T_T0 = T_T0 +1 
    
    duals[cur_step] = dual_func(alpha,y,X,lambd)
    primals[cur_step] = primal_func(w,y,X,lambd)
    if verbose: #and cur_step % n == 0 and cur_step > 0
      print("step:{},primal: {},dual: {}\n".format(cur_step,primals[cur_step], duals[cur_step] ))
  return (w_bar/T_T0, alpha_bar/T_T0 ,primals,duals)
