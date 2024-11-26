import matplotlib.pyplot as plt
import random as rdn
import numpy as np
import math as m
import scipy
import math
from scipy.optimize import minimize
from iminuit import Minuit
import data

file_path = r'tablo.csv'
tables = data.read_csv_with_multiple_tables(file_path)
# Convertir les colonnes en listes de nombres
converted_tables = data.convert_columns_to_lists(tables)

x,y = converted_tables['Tableau cuivre']
y = np.array(y)
Yerr = np.sqrt(y)


def f(x,a,b):
    return -a*x+b

def Chi2(a,b):
    chi2 = 0
    for i in range(0,len(x)):
        chi2 += math.pow(y[i]-f(x[i],a,b),2)/math.pow(Yerr[i],2)

    return chi2




minimizer = Minuit(Chi2, a=100, b=100)
minimizer.limits["a"] = (0,1000)
minimizer.limits["b"] = (0,1000)
minimizer.migrad()
a_fit = minimizer.values['a']
b_fit = minimizer.values['b']
print(minimizer) 


y_fit = []

for element in x:
    y_fit.append(f(element,a_fit,b_fit))
#print("\n incertitude sur la mesure Y= \n",Yerr)

plt.matplotlib.pyplot.errorbar(x, y, yerr = Yerr, xerr=None, fmt='r.')
plt.xlabel("temps en secondes")
plt.ylabel("nombre de coup")
plt.title("activité du Cuivre")

plt.plot(x,y_fit)
plt.show()