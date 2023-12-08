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
    x0 = x0+p*wk
    x0 = abs(x0)
    ek = np.linalg.norm(p*wk)
    k+=1
  return x0/np.sum(x0),k,f(x0,covariance_matrix)


def descente2(x0,covariance_matrix,e,method='golden_section'):
  k = 0
  ek = 2*e
  while ek>=e:
    wk = -1*grad_f(x0,covariance_matrix)
    p = pk(x0,covariance_matrix,wk,method)
    x0 = (x0+p*wk)
    x0 = abs(x0)
    ek = np.linalg.norm(p*wk)
    k+=1
  return x0/np.sum(x0),k,f(x0,covariance_matrix)



def pk(x,covariance_matrix, wk, method='golden_section'):

    if method == 'golden_section':

        alpha = golden_section_line_search(x,covariance_matrix, wk)

    else:

        alpha = wolfe_conditions_line_search(x, wk)

    return alpha

def golden_section_line_search(x,covariance_matrix, wk, c1=1e-4, max_iter=100):
    a = 0.0
    b = 1.0
    tau = 0.618

    for _ in range(max_iter):
        alpha1 = a + (1 - tau) * (b - a)
        alpha2 = a + tau * (b - a)

        f1 = f(x+ alpha1* wk,covariance_matrix)
        f2 = f(x +alpha2* wk,covariance_matrix)

        if f2 > f1:
            b = alpha2
        else:
            a = alpha1

        if abs(alpha2 - alpha1) < c1:
            break

    return (alpha1 + alpha2) / 2.0

def wolfe_conditions_line_search( x,covariance_matrix, wk, c1=1e-4, c2=0.9, max_iter=100):
    alpha = 1.0
    rho = 0.5
    phi_0 = f(x,covariance_matrix)
    phi_prime_0 = grad_f(x,covariance_matrix)*wk

    for _ in range(max_iter):
        x_next = x + alpha * wk
        phi_alpha = f(x_next,covariance_matrix)

        if phi_alpha > phi_0 + c1 * alpha * phi_prime_0:
            alpha *= rho
        else:
            phi_prime_alpha = grad_f(x_next,covariance_matrix) * wk

            if phi_prime_alpha < c2 * phi_prime_0:
                alpha *= 1.5
            else:
                return alpha

    raise ValueError("Wolfe conditions line search did not converge.")