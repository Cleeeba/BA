# %%
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore
import seaborn as sns
import concurrent.futures

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
df_2gram = spark.read.parquet("/mnt/c/Users/bincl/BA-Thesis/Dataset/2gram/warehouse/2gram_table")   

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


# %%
def buildApproximation(c1, c2, basevalue, df):
    df = df.fillna(0)
    df['approximation'] = c1 * df.iloc[:, 0] + c2 * df.iloc[:, 1] + basevalue
    return df

# %%
from pyspark.sql.functions import pandas_udf , PandasUDFType
from pyspark.sql.types import StructType, StructField, FloatType, ArrayType

# ...

# Annahme: get_pd_df, MLR, buildApproximation und andere benötigte Funktionen sind definiert

def compress_single_pandas(full, left, right):
    full = get_pd_df(full)
    left = get_pd_df(left)
    right = get_pd_df(right)

    if not (full.eq(0).all() or right.eq(0).all() or left.eq(0).all()):
        coef, intercept, dfAprox = MLR(full, left, right)
        c1, c2 = coef
        df = buildApproximation(c1, c2, intercept, dfAprox)

        dfOriginal = pd.DataFrame({'values': pd.to_numeric(full), 'zscore': zscore(full)})
        df['zscore'] = zscore(df['approximation'])

        if not df.isnull().values.any():
            sum_value = pd.to_numeric(dfOriginal['values']).sum()
            rmse = mean_squared_error(dfOriginal['zscore'], df['zscore'], squared=False)
            result = [rmse, dfOriginal['values'].to_list(), dfOriginal['zscore'].to_list(), df['approximation'].to_list(), df['zscore'].to_list()]
            return pd.DataFrame({'result': [result], 'rmse': [rmse], 'sum_value': [sum_value]})

    return pd.DataFrame({'result': [None], 'rmse': [None], 'sum_value': [None]})

# Definieren Sie das Schema für das Ergebnis der Funktion
result_schema = StructType([
    StructField("result", ArrayType(FloatType())),
    StructField("rmse", FloatType()),
    StructField("sum_value", FloatType())
])

# Erstellen Sie eine pandas_udf aus der Funktion
compress_single_pandas_udf = pandas_udf(compress_single_pandas, result_schema, PandasUDFType.SCALAR)

df = df_2gram.limit(5)
# Annahme: df ist Ihr Spark DataFrame
# Verwenden Sie die UDF auf Ihren Spark DataFrame
result_df = df.withColumn("compression_result", compress_single_pandas_udf("Frequency_N", "Frequency_L", "Frequency_R"))

# Sie können dann auf die Ergebnisspalte zugreifen, um die Werte zu extrahieren
result_df.select("compression_result.result", "compression_result.rmse", "compression_result.sum_value").show(truncate=False)

# %%
def parallel_compressWithError2gram(n, error):
    df_2_gram = df_2gram.head(n)
    result_list = []
    all_results = []
    sum_results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(compress_single, df_2_gram[i], error): i for i in range(n)}
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            try:
                result, all, sum_value = future.result()
                if result is not None:
                    result_list.extend(result)  # Use the correct variable name here
                all_results.append(all)
                sum_results.append(sum_value)
            except Exception as e:
                print(f"Error processing item {i}: {e}")

    return  result,all_results, sum_results


def compress_single(full_col, left_col, right_col):
    
    full = get_pd_df(full_col)
    left = get_pd_df(left_col)
    right = get_pd_df(right_col)
    
    if not (full.eq(0).all() or right.eq(0).all() or left.eq(0).all()):
        coef, intercept, dfAprox = MLR(full, left, right)
        c1, c2 = coef
        df = buildApproximation(c1, c2, intercept, dfAprox)
        
        dfOriginal = pd.DataFrame({'values': pd.to_numeric(full), 'zscore': zscore(full)})
        df['zscore'] = zscore(df['approximation'])
        
        if not df.isnull().values.any():
            sum_value = pd.to_numeric(dfOriginal['values']).sum()
            rmse = mean_squared_error(dfOriginal['zscore'], df['zscore'], squared=False)
            result = [rmse,dfOriginal['values'].to_list(),dfOriginal['zscore'].to_list(),df['approximation'].to_list(),df['zscore'].to_list()]
            return result, rmse, sum_value
                
          
    return None, None, None

# %%
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, FloatType, ArrayType

# Hier sollten die entsprechenden Importe für die von Ihnen verwendet
# Definieren Sie das Schema für das Ergebnis der Funktion
result_schema = StructType([
    StructField("result", ArrayType(FloatType())),
    StructField("rmse", FloatType()),
    StructField("sum_value", FloatType())
])

compress_single_udf_spark = udf(compress_single, result_schema)

# %%


from pyspark.sql.functions import col 
  
# Create a sample DataFrame 
df = df_2gram.limit(5)
  
result_df = df.withColumn("compression_result", compress_single_udf_spark("Frequency_N", "Frequency_L", "Frequency_R"))

# Sie können dann auf die Ergebnisspalte zugreifen, um die Werte zu extrahieren
result_df.select("compression_result.result", "compression_result.rmse", "compression_result.sum_value").show(truncate=False)

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


