import csv

def read_csv_with_multiple_tables(file_path):
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        tables = {}
        current_table_name = None
        current_table_data = []

        for row in reader:
            if row and row[0].startswith('Tableau'):
                if current_table_name:
                    tables[current_table_name] = current_table_data
                current_table_name = row[0]
                current_table_data = []
            elif row:  # Assurez-vous que la ligne n'est pas vide
                current_table_data.append(row)

        if current_table_name:
            tables[current_table_name] = current_table_data

    return tables

def convert_columns_to_lists(tables):
    converted_tables = {}
    for table_name, table_data in tables.items():
        # Transposer les lignes en colonnes
        transposed_data = list(map(list, zip(*table_data)))
        # Convertir chaque colonne en une liste de nombres
        converted_tables[table_name] = [list(map(int, column)) for column in transposed_data]
    return converted_tables

def print_tables(tables):
    for table_name, table_data in tables.items():
        print(f"\n{table_name}:")
        for i, column in enumerate(table_data):
            print(f"Colonne {i+1}: {column}")

# Chemin absolu du fichier CSV
file_path = r'tablo.csv'

# Lire le fichier CSV et obtenir les tableaux
tables = read_csv_with_multiple_tables(file_path)

# Convertir les colonnes en listes de nombres
converted_tables = convert_columns_to_lists(tables)
