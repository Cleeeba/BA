from itertools import chain
from pyspark import StorageLevel
from pyspark.sql import SparkSession
import pyspark.pandas as ps

spark = SparkSession.builder.appName('CorpusLoader').master('local[*]')\
.config('spark.driver.memory','100G') \
    .config("spark.sql.mapKeyDedupPolicy","LAST_WIN") \
    .config("spark.sql.adaptive.optimizeSkewedJoin.enabled", "true") \
        .config("spark.local.dir", "/mnt/simhomes/binzc/sparktemp") \
    .getOrCreate()
    

import os
import seaborn as sns
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import map_entries, explode, when, col, lit, collect_list
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import Row

from matplotlib import pyplot as plt
from pyspark.sql.functions import split, element_at, explode, map_values, array_min, broadcast, map_from_entries, arrays_zip, array_contains, monotonically_increasing_id, array_distinct, transform, arrays_zip, size, slice, collect_list, first, map_from_arrays
from pyspark.sql.types import LongType, ArrayType, IntegerType, MapType

class CorpusLoader:

    def __init__(self, root_path, spark):
        self.__root_path = root_path
        self.__spark = spark

    def load(self):
        #self.__array_df = self.__load_or_create_parquet('array.parquet', self.__create_array_df)
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
    
#cl = CorpusLoader('/mnt/c/Users/bincl/BA-Thesis/Dataset/parquets_corpus/', spark)
cl = CorpusLoader('/mnt/simhomes/binzc/data_transfer', spark)

cl.load()
#'''
# Limit für kleinere Sample size
token_array_df = cl._CorpusLoader__contains_df
#token_array_df = cl._CorpusLoader__contains_df.limit(50000)

# Baue NgramId mit child tabelle
token_array_df = token_array_df.orderBy('NgramId', 'Position').groupBy('NgramId').agg(collect_list('TokenId').alias('Tokens'))
df = token_array_df.withColumnRenamed('Tokens', 'Ngram').withColumn('Length', size('Ngram')).where('Length > 1')
df = df.withColumn('LeftChildTokenIds', slice(df.Ngram, 1, df.Length - 1))
df = df.withColumn('RightChildTokenIds', slice(df.Ngram, 2,df.Length - 1))

result = (df
          .join(token_array_df.withColumnRenamed('NgramId', 'LeftChildNgramId'), on= df.LeftChildTokenIds == token_array_df.Tokens).withColumnRenamed('Tokens', 'LCTokens')
          .join(token_array_df.withColumnRenamed('NgramId', 'RightChildNgramId'), on= df.RightChildTokenIds == token_array_df.Tokens).withColumnRenamed('Tokens', 'RCTokens')
          .select("NgramId", "LeftChildNgramId", "RightChildNgramId"))

result.printSchema()
# Frequency Table without 0
from pyspark.sql.functions import col


ngram_table = (result
              .join(cl._CorpusLoader__data_df, on=("NgramId")).withColumnRenamed("Frequency","Frequency_N").alias("og")
              .join(cl._CorpusLoader__data_df.alias("dataL"),(col("og.LeftChildNgramId") == col("dataL.NgramId"))).withColumnRenamed("Frequency","Frequency_L")
              .join(cl._CorpusLoader__data_df.alias("dataR"),(col("og.RightChildNgramId") == col("dataR.NgramId"))).withColumnRenamed("Frequency","Frequency_R")
              .select(col("og.NgramId").alias("NgramId"),"Frequency_N", "Frequency_L",col("dataL.NgramId").alias("NgramIdL"), "Frequency_R",col("dataR.NgramId").alias("NgramIdR")))

#ngram_table = ngram_table.limit(10000)
#ngram_table.printSchema()

#'''

#ngram_table.write.parquet("/mnt/c/Users/bincl/BA-Thesis/Dataset/parquets_corpus/parquets/freq", mode= 'overwrite'))
#ngram_table = spark.read.parquet("/mnt/c/Users/bincl/BA-Thesis/Dataset/parquets_corpus/parquets/freq").limit(10)
#ngram_table = spark.read.parquet("/mnt/simhomes/binzc/data_transfer/freq_small").limit(10)

from pyspark.sql import functions as F
#Frequency Table with 0
year_list = list(range(1800, 2001))

dict_years = list(chain.from_iterable(zip([F.lit(year) for year in year_list], [F.lit(0) for i in year_list])))

'''
year_df = spark.createDataFrame(list(range(1800, 2001)), schema=IntegerType())
year_df.printSchema()
df = ngram_table.crossJoin(year_df).withColumnRenamed('value', 'Year')

'''

def explode_and_filter(df, frequency_col, dict_years):
    new_col_name = f"new_{frequency_col}"
    
    result_df = df.withColumn(
        new_col_name,
        F.map_concat(
            F.create_map(*dict_years),
            frequency_col
        )
    ).select("NgramId", F.explode(new_col_name).alias("Year", "value")).filter((col("Year") >= 1800) & (col("Year") < 2000))

    return result_df

# Verwendung der Funktion für jede Tabelle
df_N = explode_and_filter(ngram_table, "Frequency_N", dict_years)
df_L = explode_and_filter(ngram_table, "Frequency_L", dict_years)
df_R = explode_and_filter(ngram_table, "Frequency_R", dict_years)


df = df_N.withColumnRenamed("value","Frequency_N").join(df_L,["NgramId","Year"])\
    .withColumnRenamed("value","Frequency_L").join(df_R,["NgramId","Year"])\
    .withColumnRenamed("value","Frequency_R").orderBy(["NgramId","Year"])

df.explain()
#df_new = df_R.withColumn("TrueFreq", col("Frequency_R").getItem("value").alias("value")).filter((col("Frequency_R").getItem("year") >= 1800) & (col("Frequency_R").getItem("year") <= 2000))


year_df = spark.createDataFrame(list(range(1800, 2001)), schema=IntegerType())
df = ngram_table.crossJoin(year_df).withColumnRenamed('value', 'Year')

df = df.withColumn("TrueFreq", when(col("Frequency_N").getItem(df.Year).isNotNull(),col("Frequency_N").getItem(df.Year)).otherwise(lit(0)))
df = df.select("NgramId","TrueFreq", "Frequency_L", "Frequency_R",'Year').withColumnRenamed('TrueFreq', 'Frequency_N')
df = df.withColumn("TrueFreq", when(col("Frequency_L").getItem(df.Year).isNotNull(),col("Frequency_L").getItem(df.Year)).otherwise(lit(0)))
df = df.select("NgramId","Frequency_N", "TrueFreq", "Frequency_R" ,'Year').withColumnRenamed('TrueFreq', 'Frequency_L')
df = df.withColumn("TrueFreq", when(col("Frequency_R").getItem(df.Year).isNotNull(),col("Frequency_R").getItem(df.Year)).otherwise(lit(0)))
df = df.select("NgramId","Frequency_N", "Frequency_L", "TrueFreq", 'Year').withColumnRenamed('TrueFreq', 'Frequency_R')
df_1 = df.select("NgramId","Frequency_N","Frequency_L","Frequency_R" )
df_1.orderBy(["NgramId","Year"])
df.explain()
exit()

df.write.partitionBy("NgramId") \
        .mode("overwrite") \
        .parquet("/mnt/simhomes/binzc/data_transfer/df_freq")


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

beta = beta.withColumnRenamed("Frequency_L","Coef_L").withColumnRenamed("Frequency_R","Coef_R")
df_1= df_1.groupby(group_column).agg(collect_list("Frequency_N").alias("Frequency_N"),collect_list("Frequency_L").alias("Frequency_L"),collect_list("Frequency_R").alias("Frequency_R"))

df = beta.join(df_1, on= "NgramId")
result_df = df.withColumn("L_multi", expr("transform(Frequency_L, x -> x * Coef_L + ic)"))
result_df = result_df.withColumn("R_multi", expr("transform(Frequency_R, x -> x * Coef_R)"))


full_df = result_df.withColumn("Aprox", expr("transform(L_multi, (x, i) -> x + R_multi[i])"))
result_df_zwischen = full_df.select("NgramId","Aprox","L_multi","R_multi")

full_df.coalesce(32)
#result_df_zwischen.show()

from pyspark.sql.functions import stddev_pop, avg, col , explode
#----------------------------------

result_df = full_df.select("NgramId","Aprox", explode("Aprox").alias("exploded_aprox"))

stats = (result_df.groupBy("NgramId")
  .agg(
      stddev_pop("exploded_aprox").alias("sd"), 
      avg("exploded_aprox").alias("avg")))


result_df= result_df.join(stats, ["NgramId"]).select("NgramId",((result_df.exploded_aprox - stats.avg) / stats.sd).alias("ZScore"))
ZScore_df= result_df.groupBy("NgramId").agg(collect_list("ZScore").alias("ZScoreArray"))
#ZScore_df.show()
# Falls fehler dann hier approx Zscore sind zu nah aneinander--------------------- Brodcast entfernt muss getestet werden ob klappt
ZScore_N_df = full_df.select("NgramId","Frequency_N",explode("Frequency_N").alias("exploded_Frequency_N"))
stats = (ZScore_N_df.groupBy("NgramId")
  .agg(
      stddev_pop("exploded_Frequency_N").alias("sd"), 
      avg("exploded_Frequency_N").alias("avg")))

ZScore_N_df= ZScore_N_df.join(stats, ["NgramId"]).select("NgramId","Frequency_N",((ZScore_N_df.exploded_Frequency_N - stats.avg) / stats.sd).alias("ZScore_N"))
ZScore_N_df= ZScore_N_df.groupBy("NgramId").agg(collect_list("ZScore_N").alias("ZScore_N_Array"))
#ZScore_N_df.show()

#----------------------------------
#result_df.show()

import pyspark.sql.functions as F

sum_df = full_df.groupBy("NgramId").agg(F.sum("Frequency_N").alias("Sum"))
'''
sum_df = full_df.select(
  "NgramId",'Frequency_N',
  F.expr('AGGREGATE(Frequency_N, 0, (acc, x) -> CAST(acc AS INT) + CAST(x AS INT))').alias('Sum')
)
'''
#sum_df.show()

from pyspark.sql.functions import col, sqrt, pow



rmse_df = ZScore_N_df.join(ZScore_df, on="NgramId")

rmse_df = rmse_df.withColumn("zip", arrays_zip("ZScoreArray", "ZScore_N_Array"))\
    .withColumn("zip", explode("zip"))\
    .select("NgramId", col("zip.ZScoreArray").alias("ZScoreArray"), col("zip.ZScore_N_Array").alias("ZScore_N_Array"))

rmse_stats = (rmse_df.groupBy("NgramId")
  .agg(
      sqrt(avg(pow(col("ZScoreArray") - col("ZScore_N_Array"), 2))).alias("rmse")))


#rmse_stats.show()

final=sum_df.join(rmse_stats, on="NgramId").join(ZScore_N_df,on="NgramId").join(ZScore_df,on="NgramId")
final= final.select("NgramId","Sum","rmse","ZScore_N_Array","ZScoreArray").persist(StorageLevel.MEMORY_AND_DISK)

#final.show()

#final.write.parquet("/mnt/simhomes/binzc/data_transfer/final_df" , mode= 'overwrite')


line_plot = final.select("ZScore_N_Array", "ZScoreArray").first()
years = list(range(1800, 2001))

pandas_df = pd.DataFrame({
    'ZScore_N_Array': line_plot.ZScore_N_Array,
    'ZScoreArray': line_plot.ZScoreArray})

plt.plot(years, pandas_df['ZScore_N_Array'], label='ZScore_N_Array')
plt.plot(years, pandas_df['ZScoreArray'], label='ZScoreArray')

plt.xlabel('Years')
plt.ylabel('Values')
plt.title('Line Plot of ZScore_N_Array and ZScoreArray over Years')
plt.legend()

plt.savefig("/mnt/simhomes/binzc/png/line_plot.png")
plt.clf()

violin = final.select("rmse")
pandas_df = violin.toPandas()

sns.violinplot(x=pandas_df["rmse"], inner_kws=dict(box_width=15, whis_width=2, color=".8"))
plt.savefig("/mnt/simhomes/binzc/png/violin_plot.png")
plt.clf()

sum_plot = final.select("Sum","rmse")
pandas_df = sum_plot.toPandas()
plt.scatter(x=pandas_df["Sum"], y=pandas_df["rmse"])
plt.xscale("log")

plt.xlabel("Sum")
plt.ylabel("RMSE")
plt.title("Scatter Plot mit logarithmischer Skala auf der x-Achse")

plt.savefig("/mnt/simhomes/binzc/png/scatter_plot.png")
plt.clf()

final.write.parquet("/mnt/simhomes/binzc/data_transfer/final_df" , mode= 'overwrite')


#beta.write.parquet("/mnt/c/Users/bincl/BA-Thesis/Dataset/parquets_corpus/parquets/param" , mode= 'overwrite')

#df = df.select(("NgramId","Frequency_N", "Frequency_L" ,"Frequency_R", "Year"))
#df.write.parquet("/mnt/simhomes/binzc/parquets_corpus/warehouse/freq" , mode= 'overwrite')



