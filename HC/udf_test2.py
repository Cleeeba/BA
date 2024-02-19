from itertools import chain
from pyspark.sql.types import *
from pyspark.sql.functions import col, count, rand, collect_list, explode, struct, count, lit, when, first,array, expr
from pyspark.sql.functions import pandas_udf, PandasUDFType

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

spark = SparkSession.builder.appName('CorpusLoader').master('local[*]')\
.config('spark.driver.memory','100G') \
    .config("spark.local.dir", "/mnt/simhomes/binzc/sparktemp") \
    .config("spark.sql.adaptive.optimizeSkewedJoin.enabled", "true") \
    .getOrCreate()
ngram_table = spark.read.parquet("/mnt/simhomes/binzc/data_transfer/freq_small")
#ngram_table = spark.read.parquet("/mnt/c/Users/bincl/BA-Thesis/Dataset/parquets_corpus/parquets/freq")
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


from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import functions as F
from pyspark.sql.functions import lit

# Create a Spark session

group_column = 'NgramId'
intercept = "ic"
y_column = 'Frequency_N'
x_columns = ['Frequency_L', 'Frequency_R']
schema = StructType([StructField('NgramId', LongType(), True), StructField('Frequency_L', DoubleType(), False), StructField('Frequency_R', DoubleType(), False),StructField('ic',  DoubleType(), True)])
#schema = df2.select(group_column, *x_columns).schema

# Beispiel DataFrame erstellen

# Lineare Regression durchführen für jede ID

grouped_data = df.groupBy(group_column).agg(collect_list("Frequency_L").alias("Frequency_L"), collect_list("Frequency_R").alias("Frequency_R"), collect_list("Frequency_R").alias("Frequency_R"))

assembler = VectorAssembler(inputCols= x_columns, outputCol="features")
df_assembled = assembler.transform(grouped_data)

lr = LinearRegression(featuresCol='features', labelCol='y_column')

model = lr.fit(df_assembled)

# Ergebnisse der Regression für jede ID sammeln


# Ergebnisse anzeige


print(model)

#beta.write.parquet("/mnt/c/Users/bincl/BA-Thesis/Dataset/parquets_corpus/test" , mode= 'overwrite')