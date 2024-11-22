import matplotlib.pyplot as plt
import random as rdn
import numpy as np
import math as m
import scipy
import math
from scipy.optimize import minimize
from iminuit import Minuit
import data


def f(x,a,b,c):
    return a*np.exp(-b*x)+c

def Chi2(a,b,c,x,y,Yerr):
    chi2 = 0
    print("avant")
    for i in range(0,len(x)-1):
        chi2 += math.pow(y[i]-f(x[i],a,b,c),2)/math.pow(Yerr[i],2)

    return chi2
