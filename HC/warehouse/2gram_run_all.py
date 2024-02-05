# %%
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore
import seaborn as sns
from alive_progress import alive_bar
from pyspark.sql.functions import monotonically_increasing_id

# %%

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import pandas as pd

start_date = 1800
end_date = 2000
numbers = list(range(start_date,end_date))

conf= SparkConf().setAll([('spark.executor.memory', '16g'), ('spark.executor.cores', '4'), ('spark.cores.max', '4'), ('spark.driver.memory','16g')])
spark = SparkSession.builder.config(conf=conf).appName('NgramSQL').getOrCreate()

#spark = SparkSession.builder.appName('3gramSQL').getOrCreate()
df_2gram = spark.read.parquet("/mnt/simhomes/binzc/parquets_corpus/warehouse/2gram_table")   

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

def buildApproximation(c1, c2, basevalue, df):
    df = df.fillna(0)
    df['approximation'] = c1 * df.iloc[:, 0] + c2 * df.iloc[:, 1] + basevalue
    return df

# %%
def compressWithError2gram(chunk_df,n,error):
    df_2_gram = chunk_df.collect()
    print("got it")
    result = []
    all = []
    sum = []
    
    with alive_bar(n,length= 20, force_tty = True, bar = 'smooth') as bar:
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
            bar()
    return result, all, sum

# %%
#result, all, sum = compressWithError2gram(10, 1)

#print(len(result))

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

total_rows = df_2gram.count()

data_result=[]
data_all= []
data_sum = []

from pyspark.sql import functions as F

sample_dict = {}
# add the partition_number as a column
df = df_2gram.withColumn('partition_num', F.spark_partition_id())

total_partition = [int(row.partition_num) for row in 
df.select('partition_num').distinct().collect()]

for each_df in total_partition:
    sample_dict[each_df] = df.where(df.partition_num == each_df) 
    

for i in range(0,len(sample_dict)):
    result, all, sum = compressWithError2gram(sample_dict[i],sample_dict[i].size(), 0.5)
    print(i)
    
    plt.close("all")
    data_result.extend(result)
    data_all.extend(all)
    data_sum.extend(sum)
    box(data_all,data_result)
    line(data_result)
    scatter(data_sum,data_all)
    
    print(len(result))
    
    
                
        


