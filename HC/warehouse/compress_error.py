from itertools import chain
import numpy
from pyspark import StorageLevel
from pyspark.sql import SparkSession
import pyspark.pandas as ps
import pandas as pd
import statsmodels.api as sm
from pyspark.sql.types import *

spark = SparkSession.builder.appName('CorpusLoader').master('local[*]')\
.config('spark.driver.memory','100G') \
    .config("spark.sql.mapKeyDedupPolicy","LAST_WIN") \
    .config("spark.sql.adaptive.optimizeSkewedJoin.enabled", "true") \
    .config("spark.local.dir", "/mnt/simhomes/binzc/sparktemp") \
    .config("spark.executor.memory", "4g")\
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")\
    .config("spark.default.parallelism", "96")\
        .getOrCreate()       
import os
import seaborn as sns
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col
from matplotlib import pyplot as plt
from pyspark.sql.functions import split, element_at, map_concat,create_map, map_from_entries, when, col, lit, collect_list,expr, arrays_zip, monotonically_increasing_id, transform, arrays_zip, size, slice, collect_list, stddev_pop, avg, col , explode, col, sqrt, pow
from pyspark.sql.types import LongType, IntegerType
from pyspark.sql import functions as f

year_list = list(range(1800, 2001))
dict_years = list(chain.from_iterable(zip([lit(year) for year in year_list], [lit(0) for i in year_list])))
group_column = 'NgramId'
intercept = "ic"
y_column = 'Frequency_N'
x_columns = ['Frequency_L', 'Frequency_R']
schema = StructType([StructField('NgramId', LongType(), True), StructField('Frequency_L', DoubleType(), False), StructField('Frequency_R', DoubleType(), False),StructField('ic',  DoubleType(), True)])


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
        parquet_path = os.path.join(os.path.join(self.__root_path, 'parquets_new'), name)
        print(parquet_path)
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
cl = CorpusLoader('/mnt/simhomes/binzc/', spark)

cl.load()
def load_ngramTable():
    
    #token_array_df = cl._CorpusLoader__contains_df
    token_array_df = cl._CorpusLoader__contains_df.repartition(6 * 128, col(group_column))
    data_parquet = cl._CorpusLoader__data_df.repartition(6 * 128, col(group_column))
    # Baue NgramId mit child tabelle
    token_array_df = token_array_df.orderBy('NgramId', 'Position').groupBy('NgramId').agg(collect_list('TokenId').alias('Tokens'))
    df = token_array_df.withColumnRenamed('Tokens', 'Ngram').withColumn('Length', size('Ngram')).where('Length > 1')
    df = df.withColumn('LeftChildTokenIds', slice(df.Ngram, 1, df.Length - 1))
    df = df.withColumn('RightChildTokenIds', slice(df.Ngram, 2,df.Length - 1))
    
    result = (df
            .join(token_array_df.withColumnRenamed('NgramId', 'LeftChildNgramId'), on= df.LeftChildTokenIds == token_array_df.Tokens).withColumnRenamed('Tokens', 'LCTokens')
            .join(token_array_df.withColumnRenamed('NgramId', 'RightChildNgramId'), on= df.RightChildTokenIds == token_array_df.Tokens).withColumnRenamed('Tokens', 'RCTokens')
            .select("NgramId", "LeftChildNgramId", "RightChildNgramId"))
    # Frequency Table without 0
    '''
    ngram_table = (result
                .join(data_parquet, on=("NgramId")).withColumnRenamed("Frequency","Frequency_N").alias("og")
                .join(data_parquet.alias("dataL"),(col("og.LeftChildNgramId") == col("dataL.NgramId"))).withColumnRenamed("Frequency","Frequency_L")
                .join(data_parquet.alias("dataR"),(col("og.RightChildNgramId") == col("dataR.NgramId"))).withColumnRenamed("Frequency","Frequency_R")
                .select(col("og.NgramId").alias("NgramId"),"Frequency_N", "Frequency_L",col("dataL.NgramId").alias("NgramIdL"), "Frequency_R",col("dataR.NgramId").alias("NgramIdR")))
    '''
    
    return result
cl = CorpusLoader('/mnt/simhomes/binzc/', spark)
from pyspark.sql.functions import coalesce
cl.load()
ngram = load_ngramTable()      
calculated_df = spark.read.parquet("/mnt/simhomes/binzc/parquets/full_final_df2").select("NgramId","rmse","Coef_R", "Coef_L","ic").repartition(4*128, "NgramId")
level2 = spark.read.parquet("/mnt/simhomes/binzc/parquets/level2_table").select("NgramId","rmse","Coef_R", "Coef_L","ic")
level3 = spark.read.parquet("/mnt/simhomes/binzc/parquets/level3_table").select("NgramId","rmse","Coef_R", "Coef_L","ic")
level4 = spark.read.parquet("/mnt/simhomes/binzc/parquets/level4_table").select("NgramId","rmse","Coef_R", "Coef_L","ic")

joined_df = calculated_df.join(level2.withColumnRenamed("rmse", "rmse_level2")
                                    .withColumnRenamed("Coef_R", "Coef_R_level2")
                                    .withColumnRenamed("Coef_L", "Coef_L_level2")
                                    .withColumnRenamed("ic", "ic_level2"), "NgramId", "left_outer") \
                         .join(level3.withColumnRenamed("rmse", "rmse_level3")
                                    .withColumnRenamed("Coef_R", "Coef_R_level3")
                                    .withColumnRenamed("Coef_L", "Coef_L_level3")
                                    .withColumnRenamed("ic", "ic_level3"), "NgramId", "left_outer") \
                         .join(level4.withColumnRenamed("rmse", "rmse_level4")
                                    .withColumnRenamed("Coef_R", "Coef_R_level4")
                                    .withColumnRenamed("Coef_L", "Coef_L_level4")
                                    .withColumnRenamed("ic", "ic_level4"), "NgramId", "left_outer")

joined_df = joined_df.withColumn("rmse", when(col("rmse_level4").isNotNull(), col("rmse_level4"))
                                       .when(col("rmse_level3").isNotNull(), col("rmse_level3"))
                                       .when(col("rmse_level2").isNotNull(), col("rmse_level2"))
                                       .otherwise(col("rmse"))) \
                     .withColumn("Coef_R", when(col("Coef_R_level4").isNotNull(), col("Coef_R_level4"))
                                          .when(col("Coef_R_level3").isNotNull(), col("Coef_R_level3"))
                                          .when(col("Coef_R_level2").isNotNull(), col("Coef_R_level2"))
                                          .otherwise(col("Coef_R"))) \
                     .withColumn("Coef_L", when(col("Coef_L_level4").isNotNull(), col("Coef_L_level4"))
                                          .when(col("Coef_L_level3").isNotNull(), col("Coef_L_level3"))
                                          .when(col("Coef_L_level2").isNotNull(), col("Coef_L_level2"))
                                          .otherwise(col("Coef_L"))) \
                     .withColumn("ic", when(col("ic_level4").isNotNull(), col("ic_level4"))
                                       .when(col("ic_level3").isNotNull(), col("ic_level3"))
                                       .when(col("ic_level2").isNotNull(), col("ic_level2"))
                                       .otherwise(col("ic")))

# Entfernen der Spalten mit Level-Präfixen
joined_df = joined_df.drop("rmse_level2", "rmse_level3", "rmse_level4",
                           "Coef_R_level2", "Coef_R_level3", "Coef_R_level4",
                           "Coef_L_level2", "Coef_L_level3", "Coef_L_level4",
                           "ic_level2", "ic_level3", "ic_level4")

# Anzeige der Ergebnisse
joined_df.select("NgramId","rmse","Coef_R", "Coef_L","ic").show()
print(calculated_df.count())
print(joined_df.count())
calculated_df = joined_df
calculated_df.write.parquet("/mnt/simhomes/binzc/parquets/full_error_table" , mode= 'overwrite')
data_parquet = cl._CorpusLoader__data_df.withColumn("MapOfString", col("Frequency").cast("string"))
#data_parquet.select("NgramId", "MapOfString").write.mode('overwrite').csv("/mnt/simhomes/binzc/results/original_data_csv" , mode= 'overwrite')
    
def compress(n):  
    
    compressed_df = calculated_df.filter(calculated_df["rmse"] <= n)
    compressed_df = compressed_df.join(ngram, on = "NgramId")
    compressed_df.show()
    compressed_df.select("NgramId","Coef_R", "Coef_L","ic", "LeftChildNgramId", "RightChildNgramId")
    #compressed_df.select("NgramId","Coef_R", "Coef_L","ic")
    compressed_df = compressed_df.withColumn("Coef_R", col("Coef_R").cast("float"))
    compressed_df = compressed_df.withColumn("Coef_L", col("Coef_L").cast("float"))
    compressed_df = compressed_df.withColumn("ic", col("ic").cast("float"))
    compressed_df.write.mode('overwrite').csv("/mnt/simhomes/binzc/results/compressed_df_csv" , mode= 'overwrite')
    
    
    rest_df = calculated_df.filter(calculated_df["rmse"] > n)
    uncompressed_df = rest_df.join(data_parquet, on= "NgramId").select("NgramId","Frequency")
   
    uncompressed_df.withColumn("MapOfString", col("Frequency").cast("string")).select("NgramId", "MapOfString").write.mode('overwrite').csv("/mnt/simhomes/binzc/results/uncompressed_df_csv")
 
    
    from pathlib import Path
    root_directory = Path("/mnt/simhomes/binzc/results/uncompressed_df_csv")
    file_size_uncompressed_df = sum(f.stat().st_size for f in root_directory.glob('**/*') if f.is_file())
    root_directory = Path("/mnt/simhomes/binzc/results/compressed_df_csv")
    file_size_compressed_df = sum(f.stat().st_size for f in root_directory.glob('**/*') if f.is_file())
    root_directory = Path("/mnt/simhomes/binzc/results/original_data_csv")
    file_size_original = sum(f.stat().st_size for f in root_directory.glob('**/*') if f.is_file())
    '''
    with open("/mnt/simhomes/binzc/results/info.txt", "a") as myfile:
        myfile.write(f""" 
                     rest datasize uncompressed : {file_size_uncompressed_df} byte
                     datasize compressed : {file_size_compressed_df} byte
                     original size : {file_size_original} byte
                     Compression rate for RMSE < {n}: {file_size_original / (file_size_compressed_df + file_size_uncompressed_df)}
                     
                     """)
    '''
    return  file_size_original / (file_size_compressed_df + file_size_uncompressed_df)  

compressionrate = []

error = numpy.arange(0, 1.5, 0.1)
for i in error:
    compressionrate.append(compress(i))  
print(compressionrate)    


plt.plot(error, compressionrate)


plt.xlabel('Rmse error')
plt.ylabel('Compression Rate')
plt.legend()

plt.savefig("/mnt/simhomes/binzc/results/cr_line_plot_csv.png")
plt.clf()

with open("/mnt/simhomes/binzc/results/CR_info.txt", "a") as myfile:
        myfile.write(f""" 
                     Error : {error} 
                     CR : {compressionrate} 
                     """)
    