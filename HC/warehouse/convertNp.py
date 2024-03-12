import csv
from itertools import chain
from pyspark import StorageLevel
from pyspark.sql import SparkSession
import pyspark.pandas as ps
import pandas as pd
import statsmodels.api as sm
from pyspark.sql.types import *
import numpy as np

'''       
df = spark.read.parquet("/mnt/simhomes/binzc/parquets/full_final_df").select("NgramId","ZScore_N_Array")
df.printSchema()
#np_full = df.toPandas().to_numpy()
np_full_zscore = df.select("ZScore_N_Array").toPandas()

np_full_zscore.to_numpy()

#np.save("/mnt/simhomes/binzc/results/full_np", np_full)
#np.savetxt("/mnt/simhomes/binzc/results/full_np", np_full, delimiter=',')
print(np_full_zscore)
np.save("/mnt/simhomes/binzc/results/zscore_np", np_full_zscore)
np.savetxt("/mnt/simhomes/binzc/results/zscore_np", np_full_zscore, fmt='%s')
'''

file_path = '/mnt/simhomes/binzc/results/zscore_np_processed.csv'
skip_lines = [7104, 17917, 20391, 21553, 33363, 56403, 69056, 73026, 90134, 103896, 105496, 106607, 151252, 162546, 166233, 177796]
data = []
output_file = '/mnt/simhomes/binzc/results/zscore_np.dat'
# Lies die Datei und extrahiere nur die gewünschten Zeilen
with open(file_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    
    for i, row in enumerate(csvreader, start=1):
        if i in skip_lines:
            continue

        # Zerlege die Zeichenkette in separate Werte
        values = [float(value) for value in row[0].split()]
        data.append(values)
        
with open(file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    with open(output_file, 'w') as dat_file:
        for row in csv_reader:
            # Hier kannst du die Logik für die Umwandlung anpassen
            # Z.B. Formatieren und Schreiben in das .dat-Format
            formatted_row = '\t'.join(row)  # Annahme: Trennzeichen ist Tabulator
            dat_file.write(formatted_row + '\n')      

# Konvertiere die Daten in ein NumPy-Array
data = np.array(data, dtype=float)
print(data)

df = pd.DataFrame(data)
df.to_csv("/mnt/simhomes/binzc/results/zscore_np.csv")
new_columns = [i for i in range(201)]

df = pd.read_csv("/mnt/simhomes/binzc/results/zscore_np.csv", header=None, skiprows=skip_lines, names=new_columns, nrows=20000, sep=",")
print(df)
df = df.iloc[3:]
print(df.dtypes)

for e in new_columns:
    df[e] = pd.to_numeric(df[e], errors='coerce')
    
df.to_csv("/mnt/simhomes/binzc/results/zscore_np_20k.csv")
np.save("/mnt/simhomes/binzc/results/zscore_np", data)
# Print the data


