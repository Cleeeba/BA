# %%
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark import SparkConf, SparkContext



spark = SparkSession.builder.master('local[4]').config('spark.driver.memory', '8g').getOrCreate()

# %%
import os
from pyspark.sql.functions import split, element_at, explode, map_values, array_min, broadcast, map_from_entries, arrays_zip, array_contains, monotonically_increasing_id, array_distinct, transform, arrays_zip, size, slice, collect_list, first, map_from_arrays
from pyspark.sql.types import LongType, ArrayType, IntegerType, MapType

class CorpusLoader:

    def __init__(self, root_path, spark):
        self.__root_path = root_path
        self.__spark = spark

    def load(self):
        self.__array_df = self.__load_or_create_parquet('array.parquet', self.__create_array_df)
        self.__token_df = self.__load_or_create_parquet('token.parquet', self.__create_token_df)
        self.__contains_df = self.__load_or_create_parquet('contains.parquet', self.__create_contains_df)
        self.__data_df = self.__load_or_create_parquet('data.parquet', self.__create_data_df)

    def __load_or_create_parquet(self, name, create_function):
        parquet_path = os.path.join(os.path.join(self.__root_path, 'parquets'), name)
        
        if not os.path.exists(parquet_path):
            print(f'File "{name}" not found. \n\t -- Creating "{name}" ...')
            
            df = create_function()
            df.write.parquet(parquet_path)

            print('\t -- Done.')

        print(f'Loading "{name}" ...')
        return self.__spark.read.parquet(parquet_path)

    def __create_token_df(self):
        one_gram_path = os.path.join(self.__root_path, '1')

        one_gram_df = spark.read.csv(one_gram_path, sep='\n', quote="").withColumnRenamed('_c0', 'Input')
        token_df = one_gram_df \
                .select(split('Input', '\t').alias('SplitInput')) \
                .select(element_at('SplitInput', 1).alias('Tokens')) \
                .select(explode(split('Tokens', ' ')).alias('Token')) \
                .orderBy('Token') \
                .withColumn('TokenId', monotonically_increasing_id()) 
        
        return token_df

    def __create_array_df(self):
        n_gram_directories = [os.path.join(self.__root_path, x) for x in os.listdir(self.__root_path) if x.isdigit()]
        
        input_df = None

        for path in n_gram_directories:
            new_input_df = spark.read.csv(path, sep='\n', quote="").withColumnRenamed('_c0', 'Input')
            
            if input_df is None:
                input_df = new_input_df
            else:
                input_df = input_df.union(new_input_df)

        split_df = input_df \
                    .select(split('Input', '\t').alias('SplitInput')) \
                    .select(element_at('SplitInput', 1).alias('Tokens'),
                            slice('SplitInput', 2, size('SplitInput')).alias('Data')) \
                    .select(split('Tokens', ' ').alias('Tokens'), 'Data')

        array_df = split_df.select('Tokens', transform('Data', lambda d: split(d, ',')).alias('Data')) \
                    .select('Tokens', transform('Data', lambda x: x[0].cast(IntegerType())).alias('Years'),
                            transform('Data', lambda x: x[1].cast(LongType())).alias('Frequency'),
                            transform('Data', lambda x: x[2].cast(LongType())).alias('BookFrequency')) \
                    .withColumn('NgramId', monotonically_increasing_id())

        return array_df

    def __create_contains_df(self):
        n_gram_df = self.__array_df

        n_gram_to_token_id_df = n_gram_df.select('NgramId', 'Tokens') \
                .select(explode('Tokens').alias('Token'), 'NgramId') \
                .join(self.__token_df, on='Token') \
                .groupBy('NgramId').agg(collect_list('TokenId').alias('TokenIds'))
        print(n_gram_to_token_id_df.count())

        contains_df = n_gram_to_token_id_df.select('NgramId', 'TokenIds') \
            .withColumn('IndexArray', transform('TokenIds', lambda x, i: i)) \
            .select('NgramId', arrays_zip('IndexArray', 'TokenIds').alias('TokenIds')) \
            .select('NgramId', explode('TokenIds').alias('TokenId')) \
            .select('NgramId', 'TokenId.IndexArray', 'TokenId.TokenIds') \
            .withColumnsRenamed({'IndexArray': 'Position', 'TokenIds': 'TokenId'}) \
            .orderBy('NgramId')
        print(contains_df.count())

        return contains_df

    ## This horrific arrays to list of structs to map construct is required, because map_from_arrays zeroes everything out.
    def __create_data_df(self):
        data_df = self.__array_df.select('NgramId', 'Years', 'Frequency', 'BookFrequency')
        data_df = data_df.withColumn('FrequencyStructs', arrays_zip('Years', 'Frequency'))
        data_df = data_df.withColumn('BookFrequencyStructs', arrays_zip('Years', 'BookFrequency'))
        data_df = data_df.withColumn('FrequencyMap', map_from_entries('FrequencyStructs'))
        data_df = data_df.withColumn('BookFrequencyMap', map_from_entries('BookFrequencyStructs'))
        data_df = data_df.select('NgramId', 'FrequencyMap', 'BookFrequencyMap')

        data_df.printSchema()
        
        return data_df.withColumnsRenamed({'FrequencyMap': 'Frequency', 'BookFrequencyMap': 'BookFrequency'})

# %%
cl = CorpusLoader('/mnt/c/Users/bincl/BA-Thesis/Dataset/parquets_corpus/', spark)

cl.load()

# %%
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import map_entries, explode, when, col, lit, collect_list
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import Row

from matplotlib import pyplot as plt

df = cl._CorpusLoader__data_df.select('NgramId', 'Frequency').limit(3)
# Create a DataFrame from the list of Row objects
year_df = spark.range(1800, 2001).toDF("id").withColumn("Year", col("id").cast(IntegerType())).select("Year")
#year_df = spark.createDataFrame(list(range(1800, 2001)), schema=IntegerType())

df = df.join(year_df).withColumnRenamed('value', 'Year')
df.printSchema()
df = df.withColumn("TrueFreq", when(col("Frequency").getItem(df.Year).isNotNull(),col("Frequency").getItem(df.Year)).otherwise(lit(0)))
df = df.select('NgramId', 'Year', 'TrueFreq').withColumnRenamed('TrueFreq', 'Frequency')
print(df.count())
df2 = df.withColumnRenamed('NgramId', 'NgramId2').withColumnRenamed('Frequency', 'Frequency2').withColumnRenamed('Year', 'Year2')
df3 = df.withColumnRenamed('NgramId', 'NgramId3').withColumnRenamed('Frequency', 'Frequency3').withColumnRenamed('Year', 'Year3')
df = df.join(df2, on=(df.Year == df2.Year2) & (df.NgramId < df2.NgramId2)).orderBy('NgramId', 'NgramId2', 'Year')
print(df.count())
df = df.join(df3, on=(df.Year == df3.Year3) & (df.NgramId2 < df3.NgramId3) & (df.NgramId < df3.NgramId3)).orderBy('NgramId', 'NgramId2', 'NgramId3', 'Year')
print(df.count())

df.show()

vectorAssembler = VectorAssembler(inputCols = ['Frequency2', 'Frequency3'], outputCol='features')
df_test = vectorAssembler.transform(df)

lr = LinearRegression(featuresCol = 'features', labelCol='Frequency', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(df_test)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

# %%
import pandas as pd
import statsmodels.api as sm
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType

group_column = 'NgramId'
intercept = "ic"
y_column = 'Frequency'
x_columns = ['Frequency2', 'Frequency3']
schema = StructType([StructField('NgramId', LongType(), True), StructField('Frequency2', DoubleType(), False), StructField('Frequency3', DoubleType(), False),StructField('ic',  DoubleType(), True)])
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
    return pd.DataFrame([[group_key] + [model.params[i] for i in x_columns] + [model.params[-1]]], columns=[group_column] + x_columns + [intercept])

df.show()
beta = df.groupby(group_column).apply(ols)
print(beta.count())
beta.show()





