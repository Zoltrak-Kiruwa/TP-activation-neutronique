import numpy as np

# Chemin vers le fichier CSV
fichier_csv = "feur.csv"

# Charger les données avec genfromtxt
data = np.genfromtxt(fichier_csv, delimiter=',', skip_header=1, filling_values=np.nan)

# Afficher les données
print(data)