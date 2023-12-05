import gzip
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

years = list(range(1400, 2021))
words = []
data_year = []
with gzip.open('C:/Users/bincl/BA-Thesis/Dataset/1_20000_nopos.gz','rt', encoding='utf-8') as input:
    for line in input:
        values = line.strip().split("\t")
        words.append(values[0]) 
        data = [entry.split(",") for entry in values[1:]]
        year = {entry[0]: entry[1] for entry in data}
        data_year.append(year)
        
df = pd.DataFrame(index=words, columns= years)      
for i, word_data in enumerate(data_year):
    word = words[i]
    for year, value in word_data.items():
        df.at[word, int(year)] = value  # Füge den Wert an der entsprechenden Position im DataFrame ein

# Fülle die NaN-Werte im DataFrame mit 0
df.fillna(0, inplace=True)
print(df) 

numpy_array = df.values

# Das Numpy-Array enthält die Werte aus dem DataFrame
print(numpy_array)

#np.save('Uf48.bin.dat', numpy_array)
with open('1GramBooks_TEST.tsv', 'wb') as file:
    pickle.dump(numpy_array, file)
