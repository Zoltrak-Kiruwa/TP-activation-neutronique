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

a=686.23287508; b=3e-3; c=0

xb,yb = converted_tables['Tableau seuil = -60mV'] ; xb = np.array(xb) ; yb = np.array(yb)

x,y = converted_tables['Tableau vanadium'] ; x = np.array(x) ;y = np.array(y)

moy = np.mean(yb)
sigma = np.std(yb)

yb = np.array(yb) ; Yb = np.array([])

yb_gen=np.random.normal(moy,sigma,abs(len(x)-len(yb))) ; yb = np.concatenate((yb,yb_gen))

#ajustement central réduit

for i in range(len(yb)):
    Yb = np.append(Yb,(yb[i]-moy)/sigma)
    
Y =[]
for i in range(len(x)):
    Y = np.append(Y,(y[i]-moy)/sigma)
    
Y_ = pure_sig.pure_signal(Y,yb)


def f(x,a,b,k):
    return a*np.exp(-b*x)

def Chi2(a,b,k):
    chi2 = 0
    for i in range(int(k),len(x)):
        
        chi2 += math.pow(Y_[i]-f(x[i],a,b,c),2)/math.pow(np.sqrt(Y_[i]),2)

    return chi2

def find_best_lambda(a_,b_):

    list_a = np.array([])
    list_b = np.array([])
    list_c = np.array([])
    list_minimizer = np.array([])
    

    for i in range(20,60):
        
        minimizer = Minuit(Chi2,a=a_,b=b_,k=i)
        minimizer.limits["a"] = (0,1000)
        minimizer.limits["b"] = (0,0.01)
        minimizer.fixed["k"] = True
        minimizer.migrad()
        a_fit = minimizer.values['a']
        b_fit = minimizer.values['b']
        k_fit = minimizer.values['k']
        list_a = np.append(list_a,a_fit)
        list_b = np.append(list_b,b_fit)

        list_minimizer = np.append(list_minimizer,minimizer)

    arg = np.argmin(list_b)
    print(np.min(list_b))
    return list_a[arg],list_b[arg],list_minimizer[arg],arg


    
a_fit,b_fit,Minimizer,arg = find_best_lambda(a,b)    

print("a = ",a_fit,"b_fit = ",b_fit,"arg = ",20+arg)
y_fit = []
print(Minimizer)

for element in x:
    y_fit.append(f(element,a_fit,b_fit))
#print("\n incertitude sur la mesure Y= \n",Yerr)

plt.plot(x,Y_,'r.',label="donnée centré réduit")
plt.plot(x,y_fit,label="fit du vanadium, "+r" $\lambda$"+" = "+str(b_fit))
plt.xlabel("temps en secondes")
plt.ylabel("nombre de coup")
plt.title("activité du vanadium")
plt.legend(fontsize='5')
plt.show()