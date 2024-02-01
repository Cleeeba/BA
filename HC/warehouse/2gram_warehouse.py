from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('CorpusLoader').master('local[4]').config('spark.driver.memory', '8g').getOrCreate()

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
    
cl = CorpusLoader('/mnt/c/Users/bincl/BA-Thesis/Dataset/parquets_corpus/', spark)

cl.load()

token_array_df = cl._CorpusLoader__contains_df.orderBy('NgramId', 'Position').groupBy('NgramId').agg(collect_list('TokenId').alias('Tokens'))
df = token_array_df.withColumnRenamed('Tokens', 'Ngram').withColumn('Length', size('Ngram')).where('Length > 1')
#df.cache()

df = df.withColumn('LeftChildTokenIds', slice(df.Ngram, 1, df.Length - 1))
df = df.withColumn('RightChildTokenIds', slice(df.Ngram, 2,df.Length - 1))

result = df.join(token_array_df.withColumnRenamed('NgramId', 'LeftChildNgramId'), on=df.LeftChildTokenIds == token_array_df.Tokens).withColumnRenamed('Tokens', 'LCTokens')

table = result.join(token_array_df.withColumnRenamed('NgramId', 'RightChildNgramId'), on=df.RightChildTokenIds == token_array_df.Tokens).withColumnRenamed('Tokens', 'RCTokens')

result = table.select("NgramId","LeftChildNgramId","RightChildNgramId")

from pyspark.sql.functions import col
#1000 limit is 3m 30
result = result.join(cl._CorpusLoader__data_df, on=("NgramId")).withColumnRenamed("Frequency","Frequency_N")
result = result.join(cl._CorpusLoader__data_df.alias("dataL"),(col("LeftChildNgramId") == col("dataL.NgramId"))).withColumnRenamed("Frequency","Frequency_L")
ngram_table = result.join(cl._CorpusLoader__data_df.alias("dataR"),(col("LeftChildNgramId") == col("dataR.NgramId"))).withColumnRenamed("Frequency","Frequency_R")
ngram_table= ngram_table.select("Frequency_N","Frequency_L","Frequency_R")

ngram_table.write.parquet("/mnt/c/Users/bincl/BA-Thesis/Dataset/2gram/warehouse/2gram_table" , mode= 'overwrite')



