from pyspark.sql.types import *
from pyspark.sql.functions import col, count, rand, collect_list, explode, struct, count, lit
from pyspark.sql.functions import pandas_udf, PandasUDFType

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

spark = SparkSession.builder.appName('NgramSQL').config('spark.driver.memory','32G').config('spark.executor.memory','4G').getOrCreate()


df = spark.range(0, 10 * 1000 * 10000).withColumn('id', (col('id') / 10000).cast('integer')).withColumn('v', rand())
df.repartition(10)
df.cache()
df.count()
df.show()

import pandas as pd
import statsmodels.api as sm

df2 = df.withColumn('y', rand()).withColumn('x1', rand()).withColumn('x2', rand()).select('id', 'y', 'x1', 'x2')
print(df2.count())
df2.show()

# df has four columns: id, y, x1, x2

group_column = 'id'
intercept = "ic"
y_column = 'y'
x_columns = ['x1', 'x2']
schema = df2.select(group_column, *x_columns).schema
print(schema)

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
# Input/output are both a pandas.DataFrame
def ols(pdf):
    group_key = pdf[group_column].iloc[0]
    y = pdf[y_column]
    X = pdf[x_columns]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return pd.DataFrame([[group_key] + [model.params[i] for i in x_columns]], columns=[group_column] + x_columns)

#df2.show()
#beta = df2.groupby(group_column).apply(ols)
#print(beta.count())

data = [("A", 10, 3),
        ("A", 15, 2),
        ("B", 5, 1),
        ("B", 8, 2),
        ("B", 12, 1)]

data_c = [("A", 2, 1, 7),
        ("B", 3, 2, 6),]

columns = ["NgramId", "Frequency_L", "Frequency_R"]
columns_c = ["NgramId", "Coef_L", "Coef_R", "ic"]

beta = spark.createDataFrame(data, columns)
df = spark.createDataFrame(data_c, columns_c)

from pyspark.sql.window import Window
from pyspark.sql.functions import first

df = beta.join(df, on = "NgramId")
df.printSchema()
window_spec = Window.partitionBy('NgramId')
transformed_df = df.withColumn(
    'L_multi',
    col('Frequency_L') * first('Coef_L').over(window_spec) + first('ic').over(window_spec)
)
result_df = transformed_df.withColumn(
    'R_multi',
    col('Frequency_R') * first('Coef_R').over(window_spec)
)
aprox_table = result_df.withColumn(
    'Aprox',
    col('L_multi') + col('R_multi')
)

aprox_table.show()

#beta.write.parquet("/mnt/c/Users/bincl/BA-Thesis/Dataset/parquets_corpus/test" , mode= 'overwrite')