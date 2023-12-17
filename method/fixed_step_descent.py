import numpy as np

# def f(w,covariance_matrix):
#   return np.matmul(np.matmul(np.transpose(w),covariance_matrix),w)

# def grad_f(w,covariance_matrix):
#   return np.matmul(2*np.transpose(w),covariance_matrix)


def f(w, covariance_matrix):
    """
    Portfolio risk objective function.
    
    Parameters:
        w (numpy array): Portfolio weights.
        covariance_matrix (numpy array): Covariance matrix of asset returns.
    
    Returns:
        float: Portfolio risk.
    """
    total_sum = np.sum(np.exp(w))
    weights = np.exp(w) / total_sum
    portfolio_risk = 1 / total_sum**2 * np.sum(np.outer(weights, weights) * covariance_matrix)
    
    return portfolio_risk

def grad_f(w, covariance_matrix):
    """
    Gradient of the portfolio risk objective function.
    
    Parameters:
        w (numpy array): Portfolio weights.
        covariance_matrix (numpy array): Covariance matrix of asset returns.
    
    Returns:
        numpy array: Gradient of the portfolio risk.
    """
    total_sum = np.sum(np.exp(w))
    weights = np.exp(w) / total_sum
    
    first_term = -2 * np.exp(w) / total_sum**3 * np.sum(np.outer(np.exp(w), np.exp(w)) * covariance_matrix)
    
    second_term = 2 * np.exp(w) * np.sum(np.outer(weights, covariance_matrix @ weights))
    
    gradient = first_term + second_term
    
    return gradient

def descente_markowitz(x0,covariance_matrix,e,p, max_iter=10000):
  """ 
  Gradient descent algorithm to minimize the portfolio risk.

  Parameters:
      x0 (numpy array): Initial portfolio weights.
      covariance_matrix (numpy array): Covariance matrix of asset returns.
      e (float): Tolerance for the stopping criterion.
      p (float): Step size.
      max_iter (int): Maximum number of iterations.

  Returns:
      numpy array: Optimal portfolio weights.
      int: Number of iterations.
      float: Portfolio risk.
  """
  k = 0
  ek = 2*e
  while ek>=e and k<max_iter:
    wk = -1*grad_f(x0,covariance_matrix)
    x0 = x0+p*wk
    ek = np.linalg.norm(p*wk)
    k+=1
  return np.exp(x0)/np.sum(np.exp(x0)),k,f(x0,covariance_matrix)


def descente2(x0,covariance_matrix,e,method='golden_section', max_iter=10000):
  """
  Golden section algorithm to minimize the portfolio risk.

  Parameters:
      x0 (numpy array): Initial portfolio weights.
      covariance_matrix (numpy array): Covariance matrix of asset returns.
      e (float): Tolerance for the stopping criterion.
      method (str): Line search method.
      max_iter (int): Maximum number of iterations.
    
  Returns:
      numpy array: Optimal portfolio weights.
      int: Number of iterations.
      float: Portfolio risk.
  """
  k = 0
  ek = 2*e
  while ek>=e and k<max_iter:
    wk = -1*grad_f(x0,covariance_matrix)
    p = pk(x0,covariance_matrix,wk,method)
    x0 = (x0+p*wk)
    ek = np.linalg.norm(p*wk)
    k+=1
  return np.exp(x0)/np.sum(np.exp(x0)),k,f(x0,covariance_matrix)



def pk(x,covariance_matrix, wk, method='golden_section'):
    """
    Line search to find the optimal step size.

    Parameters:
        x (numpy array): Portfolio weights.
        covariance_matrix (numpy array): Covariance matrix of asset returns.
        wk (numpy array): Negative gradient of the portfolio risk.
        method (str): Line search method.

    Returns:
        float: Optimal step size.
    """


    if method == 'golden_section':

        alpha = golden_section_line_search(x,covariance_matrix, wk)

    else:

        alpha = wolfe_conditions_line_search(x, wk)

    return alpha

def golden_section_line_search(x,covariance_matrix, wk, c1=1e-4, max_iter=100):
    """
    Golden section line search to find the optimal step size.
    
    Parameters:
        x (numpy array): Portfolio weights.
        covariance_matrix (numpy array): Covariance matrix of asset returns.
        wk (numpy array): Negative gradient of the portfolio risk.
        c1 (float): Parameter for the stopping criterion.
        max_iter (int): Maximum number of iterations.
        
        Returns:
        float: Optimal step size.
        """
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