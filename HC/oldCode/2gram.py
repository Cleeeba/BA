# %%
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore
import gzip
import os
import seaborn as sns
import csv
#from alive_progress import alive_bar
from collections import defaultdict
import time
import cProfile

# %%

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
#import pandas as pd

start_date = 1800
end_date = 2000
numbers = list(range(start_date,end_date))

conf= SparkConf().setAll([('spark.executor.memory', '16g'), ('spark.executor.cores', '4'), ('spark.cores.max', '4'), ('spark.driver.memory','16g')])
spark = SparkSession.builder.config(conf=conf).appName('NgramSQL').getOrCreate()

#spark = SparkSession.builder.appName('3gramSQL').getOrCreate()
df_2gram = spark.read.parquet("/mnt/c/Users/bincl/BA-Thesis/Dataset/2gram/warehouse/2gram_table")   

# %%
def get_pd_df_old(df):
    matched_pandas_df = pd.Series(df)
    matched_pandas_df.sort_index(inplace = True)
    matched_pandas_df.index.astype('int')
    matched_pandas_df = matched_pandas_df.reindex(numbers, fill_value= 0)
    return matched_pandas_df


# %%
def get_pd_df(df):
    matched_pandas_df = pd.Series(df).astype(int).reindex(numbers, fill_value=0)
    return matched_pandas_df

# %%
def MLR(full,left,right):
    X = pd.concat([left,right],axis=1)
    reg = LinearRegression()
    reg.fit(X, full) 
    return reg.coef_,reg.intercept_,X

def buildApproximation_old(c1,c2,basevalue,df):
    df = df.astype('float')
    
    df = df.fillna(0)
    
    df['scaledFirst'] = df.iloc[:,0].apply(lambda x: x * c1) 
    df['scaledLast'] = df.iloc[:,1].apply(lambda x: x * c2) 
    df['approximation'] = df['scaledFirst'] + df['scaledLast'] + basevalue
    return df

# %%
def buildApproximation(c1, c2, basevalue, df):
    df = df.fillna(0)
    df['approximation'] = c1 * df.iloc[:, 0] + c2 * df.iloc[:, 1] + basevalue
    return df

# %%
def compressWithError2gram(n,error):
    df_2_gram = df_2gram.head(n)
    result = []
    all = []
    sum = []
    
    #with alive_bar(n,length= 20, force_tty = True, bar = 'smooth') as bar:
    for i in range(n):
            df_file = df_2_gram[i]
            full = get_pd_df(df_file['Frequency_N'])
            left = get_pd_df(df_file['Frequency_L'])
            right =  get_pd_df(df_file['Frequency_R'])
            if not(full.eq(0).all() or right.eq(0).all() or left.eq(0).all()):
                coef,intercept,dfAprox = MLR(full,left,right)
                c1,c2 = coef
                df = buildApproximation(c1,c2,intercept,dfAprox)
                dfOriginal = pd.DataFrame({'values': pd.to_numeric(full), 'zscore': zscore(full)})
                df['zscore'] = zscore(df['approximation'])
                if not(df.isnull().values.any()):
                    sum.append(pd.to_numeric(dfOriginal['values']).sum()) 
                    rmse = mean_squared_error(dfOriginal['zscore'], df['zscore'], squared = False)
                    if rmse <= error:
                        result.append([rmse,dfOriginal['values'],dfOriginal['zscore'],df['approximation'],df['zscore']]) 
                    all.append(rmse)    
                #bar()
    return result, all, sum


# %%
#cProfile.run('compressWithError2gram(1000, 1)', sort = 'cumulative')
result, all, sum = compressWithError2gram(1000, 1)

# %%
def box(all,result):
    n = 0
    plt.boxplot(all)
    
    plt.show()
    rmse_with_error = []
    for i in result:
        rmse_with_error.append(i[0])
    plt.boxplot(rmse_with_error)
    plt.show()



    sns.violinplot(x= rmse_with_error, inner="point")
    plt.show()

# %%
def line(result):
    n=0
    result[n][1].plot()
    result[n][3].plot()
    plt.show()
    result[n][2].plot()
    result[n][4].plot()
    plt.show()
    print(result[n][0])

# %%
def scatter(sum, all):
    sns.violinplot(x= all, inner_kws=dict(box_width=15, whis_width=2, color=".8"))
    plt.show()
    plt.scatter(sum, all)
    plt.xscale('log')
    plt.show()

# %%
data_result=[]
data_all= []
data_sum = []
data_result.extend(result)
data_all.extend(all)
data_sum.extend(sum)
box(data_all,data_result)
line(data_result)
scatter(data_sum,data_all)


