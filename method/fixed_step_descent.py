def descente1(grad_f,x0,e,p):
  k = 0
  ek = 2*e
  while ek>=e:
    wk = -1*grad_f(x0)
    ek = (x0+p*wk-x0).norm()
    x0 = x0+p*wk
    k+=1
  return x0,k