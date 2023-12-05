import numpy as np

def f(w,covariance_matrix):
  return np.matmul(np.matmul(np.transpose(w),covariance_matrix),w)

def grad_f(w,covariance_matrix):
  return np.matmul(2*np.transpose(w),covariance_matrix)

def descente_markowitz(x0,covariance_matrix,e,p):
  k = 0
  ek = 2*e
  while ek>=e :
    wk = -1*grad_f(x0,covariance_matrix)
    ek = np.linalg.norm(x0+p*wk-x0)
    x0 = x0+p*wk
    k+=1
  return x0