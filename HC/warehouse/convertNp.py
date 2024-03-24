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

import numpy as np
from matplotlib import pyplot as plt

path = "C:/Users/bincl/Desktop/zscore_np_20k.csv"
data = np.genfromtxt(path, delimiter=',', dtype=np.float32)
#arr = np.fromfile(path)
print(data.dtype)
print(data.shape)
print(data)
#arr = arr.astype(np.float32)
data.tofile('C:/Users/bincl/Desktop/zscore_np_20k.npy')

arr = np.fromfile("C:/Users/bincl/Desktop/zscore_np_20k.npy", dtype=np.float32)
print(arr.dtype)
print(arr.shape)
print(arr)
##arr2 = np.fromfile('/mnt/simhomes/binzc/results/zscore_np_20k.npy').reshape((19998, 201))

#print(arr2 - arr)
