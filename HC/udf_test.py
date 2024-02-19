from pyspark.sql.types import *
from pyspark.sql.functions import col, count, rand, collect_list, explode, struct, count, lit, when
from pyspark.sql.functions import pandas_udf, PandasUDFType

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

spark = SparkSession.builder.appName('NgramSQL').config('spark.driver.memory','32G').config('spark.executor.memory','4G').getOrCreate()
sc = spark.sparkContext

ngram_table = spark.read.parquet("/mnt/c/Users/bincl/BA-Thesis/Dataset/parquets_corpus/parquets/freq")
year_df = spark.createDataFrame(list(range(1800, 2001)), schema=IntegerType())
df = ngram_table.crossJoin(year_df).withColumnRenamed('value', 'Year')
#df.printSchema()


df = df.withColumn("TrueFreq", when(col("Frequency_N").getItem(df.Year).isNotNull(),col("Frequency_N").getItem(df.Year)).otherwise(lit(0)))
df = df.select("NgramId","TrueFreq", "Frequency_L", "Frequency_R",'Year').withColumnRenamed('TrueFreq', 'Frequency_N')

df = df.withColumn("TrueFreq", when(col("Frequency_L").getItem(df.Year).isNotNull(),col("Frequency_L").getItem(df.Year)).otherwise(lit(0)))
df = df.select("NgramId","Frequency_N", "TrueFreq", "Frequency_R" ,'Year').withColumnRenamed('TrueFreq', 'Frequency_L')
df = df.withColumn("TrueFreq", when(col("Frequency_R").getItem(df.Year).isNotNull(),col("Frequency_R").getItem(df.Year)).otherwise(lit(0)))
df = df.select("NgramId","Frequency_N", "Frequency_L", "TrueFreq", 'Year').withColumnRenamed('TrueFreq', 'Frequency_R')
df.printSchema()
df_1 = df.select("NgramId","Frequency_N", "Year","Frequency_L","Frequency_R" )

#df_1.show()

#UDF für MLR
import pandas as pd

import statsmodels.api as sm
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType

group_column = 'NgramId'
intercept = "ic"
y_column = 'Frequency_N'
x_columns = ['Frequency_L', 'Frequency_R']
schema = StructType([StructField('NgramId', LongType(), True), StructField('Frequency_L', DoubleType(), False), StructField('Frequency_R', DoubleType(), False),StructField('ic',  DoubleType(), True)])
#schema = df2.select(group_column, *x_columns).schema
print(schema)

#@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
# Input/output are both a pandas.DataFrame
import pyspark.pandas as ps
def ols(pdf):
    group_key = pdf[group_column].iloc[0]
    y = pdf[y_column]
    X = pdf[x_columns]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return ps.DataFrame([[group_key] + [model.params[i] for i in x_columns] + [model.params['const']]], columns=[group_column] + x_columns + [intercept])

from pyspark.sql.functions import expr
beta = df_1.groupby(group_column).applyInPandas(ols,schema= schema)
beta.drop("Frequency_N")
beta.explain()

beta = beta.withColumnRenamed("Frequency_L","Coef_L").withColumnRenamed("Frequency_R","Coef_R")
df_1= df_1.groupby(group_column).agg(collect_list("Frequency_N").alias("Frequency_N"),collect_list("Frequency_L").alias("Frequency_L"),collect_list("Frequency_R").alias("Frequency_R"))

df = beta.join(df_1, on= "NgramId")
result_df = df.withColumn("L_multi", expr("transform(Frequency_L, x -> x * Coef_L + ic)"))
result_df = result_df.withColumn("R_multi", expr("transform(Frequency_R, x -> x * Coef_R)"))


full_df = result_df.withColumn("Aprox", expr("transform(L_multi, (x, i) -> x + R_multi[i])"))
result_df_zwischen = full_df.select("NgramId","Aprox","L_multi","R_multi")
result_df_zwischen.explain()


#beta.write.parquet("/mnt/c/Users/bincl/BA-Thesis/Dataset/parquets_corpus/test" , mode= 'overwrite')