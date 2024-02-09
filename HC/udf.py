from pyspark.sql.types import *
from pyspark.sql.functions import col, count, rand, collect_list, explode, struct, count, lit
from pyspark.sql.functions import pandas_udf, PandasUDFType

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

spark = SparkSession.builder.appName('NgramSQL').config('spark.driver.memory','32G').config('spark.executor.memory','4G').getOrCreate()


df = spark.range(0, 10 * 1000 * 1000).withColumn('id', (col('id') / 10000).cast('integer')).withColumn('v', rand())
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
schema = StructType([StructField('id', IntegerType(), True), StructField('x1', DoubleType(), False), StructField('x2', DoubleType(), False),StructField('ic',  DoubleType(), True)])
#schema = df2.select(group_column, *x_columns).schema
print(schema)

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
# Input/output are both a pandas.DataFrame
def ols(pdf):
    group_key = pdf[group_column].iloc[0]
    y = pdf[y_column]
    X = pdf[x_columns]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    df['approximation'] = model.params[0] * X.iloc[:, 0] + model.params[0] * X.iloc[:, 1] + model.params['const']
    return pd.DataFrame([[group_key] + [model.params[i] for i in x_columns] + [model.params['const']]], columns=[group_column] + x_columns + [intercept])

df2.show()
beta = df2.groupby(group_column).apply(ols).apply()
print(beta.count())
#beta.write.parquet("/mnt/c/Users/bincl/BA-Thesis/Dataset/parquets_corpus/test" , mode= 'overwrite')