import numpy as np

# y= 0.01x^2 + 0.1x
def function_1(x):
    return 0.01*x**2 + 0.1*x

def numerical_diff(f, x):
    h= 1e-4
    return (f(x+h)- f(x-h))/ 2*h

def function_tmp1(x0):
    return x0**2 + 4**2

def numerical_gradient(f, x):
    h= 1e-4
    grad= np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val= x[idx]
        x[idx]= tmp_val+ h
        fxh1= f(x)
        
        x[idx]= tmp_val- h
        fxh2= f(x)
        
        grad[idx]= (fxh1- fxh2) /(2*h)
        x[idx]= tmp_val # 원래 값 복원
    return grad

def graient_descent(f, init_x, lr= 0.1, step_num= 100):
    x= init_x
    
    for i in range(step_num):
        grad= numerical_gradient(f, x);
        x-= lr*grad
    return x

def function_2(x):
    return x[0]**2 + x[1]**2

# init_x= np.array([-3.0, 4.0])
# print(graient_descent(function_2, init_x=init_x, lr= 0.1, step_num=100))


