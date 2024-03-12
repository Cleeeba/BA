# %%
import pywt
import numpy as np
from PIL import Image
import pandas as pd
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error



def dwt_compression(data,threshold):
   
    coeffs_r = pywt.dwt2(data, 'haar')
   
    coeffs = list(coeffs_r)
    coeffs = tuple((pywt.threshold(c, threshold, mode='soft') for c in coeffs))
    return coeffs


def dwt_decompress(coeff_dwt):
    return pywt.idwt2(coeff_dwt, 'haar')

# %%
def utf8len(s):
    return len(s.encode('utf-8'))

# %%
#df = pd.read_csv('/mnt/simhomes/binzc/results/1gram.csv')
df = pd.read_csv('/mnt/simhomes/binzc/results/zscore_np_20k.csv')
data=df.to_numpy()
compressed = dwt_compression(data)
print(compressed)
print(len(compressed))

# %%
decompressed= dwt_decompress(compressed)


# %%
zscore_original = np.array(zscore(data))
zscore_decompressed = np.array(zscore(decompressed))
     
print("original array") 
print(zscore_original)   
print(zscore_original.shape) 
print("decompressed array") 
print(zscore_decompressed)   
print(zscore_decompressed.shape) 

min_rows, min_cols = min(zscore_original.shape[0], zscore_decompressed.shape[0]), min(zscore_original.shape[1], zscore_decompressed.shape[1])
zscore_original = zscore_original[:min_rows, :min_cols]
zscore_decompressed = zscore_decompressed[:min_rows, :min_cols]

rmse = mean_squared_error(zscore_original, zscore_decompressed, squared = False)

# %%
c_data=compressed

import csv

# Annahme: c_data ist Ihr Tupel mit Daten
# Beispiel: c_data = ([1, 2, 3], [4, 5, 6])
#print(compressed)
# Öffnen Sie die CSV-Datei zum Schreiben
csv_file_path = "/mnt/simhomes/binzc/results/dwt_compressed_20k.csv"
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    for array in compressed:
        csv_writer.writerow(array)


from pathlib import Path

file_size_compressed_df = Path("/mnt/simhomes/binzc/results/dwt_compressed_20k.csv").stat().st_size
file_size_original = Path('/mnt/simhomes/binzc/results/zscore_np_20k.csv').stat().st_size

print("CR")
print(file_size_original / file_size_compressed_df) 
print("RMSE")
print(rmse)
