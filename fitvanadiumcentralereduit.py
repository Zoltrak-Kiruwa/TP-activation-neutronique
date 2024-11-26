import matplotlib.pyplot as plt
import random as rdn
import numpy as np
import math as m
import scipy
import math
from scipy.optimize import minimize
from iminuit import Minuit
import data
import pure_sig

file_path = r'tablo.csv'
tables = data.read_csv_with_multiple_tables(file_path)
# Convertir les colonnes en listes de nombres
converted_tables = data.convert_columns_to_lists(tables)



def f(x,a,b,c):
    return a*np.exp(-b*x)+c

def Chi2(a,b,c):
    chi2 = 0
    for i in range(40,len(x)):
        chi2 += math.pow(Y_[i]-f(x[i],a,b,c),2)/math.pow(np.sqrt(Y_[i]),2)

    return chi2

xb,yb = converted_tables['Tableau seuil = -60mV']

xb = np.array(xb)
yb = np.array(yb)

x,y = converted_tables['Tableau vanadium']
x = np.array(x)
y = np.array(y)

moy = np.mean(yb)
sigma = np.std(yb)

yb = np.array(yb)
Yb = np.array([])

yb_gen=np.random.normal(moy,sigma,abs(len(x)-len(yb)))
yb = np.concatenate((yb,yb_gen))

#ajustement central réduit

for i in range(len(yb)):
    Yb = np.append(Yb,(yb[i]-moy)/sigma)
    
Y =[]
for i in range(len(x)):
    Y = np.append(Y,(y[i]-moy)/sigma)
    
Y_ = pure_sig.pure_signal(Y,yb)
    
print("size x = ",len(Y_))
Yerr = np.sqrt(abs(Y_))

print(Y_)

minimizer = Minuit(Chi2, a=686.23287508, b=3e-3, c=0)
minimizer.limits["a"] = (0,1000)
minimizer.limits["b"] = (0,0.1)
minimizer.limits["c"] = (0,0)
minimizer.migrad()
a_fit = minimizer.values['a']
b_fit = minimizer.values['b']
c_fit = minimizer.values['c']
print(minimizer) 


y_fit = []

for element in x:
    y_fit.append(f(element,a_fit,b_fit,c_fit))
#print("\n incertitude sur la mesure Y= \n",Yerr)

plt.plot(x,Y_,'r.')
plt.xlabel("temps en secondes")
plt.ylabel("nombre de coup")
plt.title("activité du vanadium")

plt.plot(x,y_fit)
plt.show()