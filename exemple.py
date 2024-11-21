import csv

def read_csv_and_convert_columns(file_path):
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        table_name = None
        data = []

        for row in reader:
            if row and row[0].startswith('Tableau'):
                table_name = row[0]
            elif row:  # Assurez-vous que la ligne n'est pas vide
                data.append(list(map(int, row)))

    # Transposer les lignes en colonnes
    transposed_data = list(map(list, zip(*data)))

    return table_name, transposed_data

def print_columns(table_name, columns):
    print(f"\n{table_name}:")
    for i, column in enumerate(columns):
        print(f"Colonne {i+1}: {column}")

# Chemin absolu du fichier CSV
file_path = r'C:\Users\Redouane\OneDrive\Bureau\notesTP\codeTP\data.csv'

# Lire le fichier CSV et convertir les colonnes en listes de nombres
table_name, columns = read_csv_and_convert_columns(file_path)

# Afficher les colonnes converties
print_columns(table_name, columns)