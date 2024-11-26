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

x,y = converted_tables['Tableau argent'] ; x = np.array(x) ;y = np.array(y)

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


def f(x,a,b,c):
    return a*np.exp(-b*x)+c

def Chi2(a,b,c,k,end):
    chi2 = 0
    for i in range(int(k),int(end)):
        
        chi2 += math.pow(yl[i]-f(x[i],a,b,c),2)/math.pow(np.sqrt(yl[i]),2)

    return chi2

def find_best_lambda(a_,b_,c_,start,finish,endfit):

    list_a = np.array([])
    list_b = np.array([])
    list_c = np.array([])
    list_minimizer = np.array([])
    

    for i in range(start,finish):
        
        minimizer = Minuit(Chi2,a=a_,b=b_,c=c_,k=i,end=endfit)
        minimizer.limits["a"] = (0,1000)
        minimizer.limits["b"] = (0,0.01)
        minimizer.limits["c"] = (0,0)
        minimizer.fixed["k"] = True
        minimizer.fixed["end"] = True
        minimizer.migrad()
        a_fit = minimizer.values['a']
        b_fit = minimizer.values['b']
        c_fit = minimizer.values['c']
        k_fit = minimizer.values['k']
        list_a = np.append(list_a,a_fit)
        list_b = np.append(list_b,b_fit)
        list_c = np.append(list_c,c_fit)
        list_minimizer = np.append(list_minimizer,minimizer)

    arg = np.argmin(list_b)
    print(np.min(list_b))
    return list_a[arg],list_b[arg],list_c[arg],list_minimizer[arg],arg

#fit argent 108
yl = Y_
a_fit,b_fit,c_fit,Minimizer,arg = find_best_lambda(a,b,c,20,60,len(x))   

print("a = ",a_fit,"b_fit = ",b_fit,"c = ",c_fit,"arg = ",20+arg)
y_fit = np.array([])
print("Argent 108 :",Minimizer)
print("\n")


for element in x:
    y_fit=np.append(y_fit,f(element,a_fit,b_fit,c_fit))
    
#fit Argent 110
curve_diff = np.array([])
for i in range(0,len(x)):
    curve_diff = np.append(curve_diff,Y_[i]-y_fit[i])
    
yl = curve_diff

a_fit1,b_fit1,c_fit1,Minimizer1,arg1 = find_best_lambda(200,0.1,0,0,1,30)
print("Argent 110 :",Minimizer1)
print("\n")

y_fit1 = np.array([])

for element in x:
    y_fit1=np.append(y_fit1,f(element,a_fit1,b_fit1,c_fit1))

#print("\n incertitude sur la mesure Y= \n",Yerr)

reverse = np.array([])
for i in range(len(x)):
    reverse = np.append(reverse,y_fit[i]+y_fit1[i])

plt.plot(x,Y_,'r.')
plt.xlabel("temps en secondes")
plt.ylabel("nombre de coup")
plt.title("activité du argent")
plt.plot(x,y_fit,'b-')
plt.plot(x,y_fit1,'y-')
plt.plot(x,reverse,color = 'purple')
plt.show()