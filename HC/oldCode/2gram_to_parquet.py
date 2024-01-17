from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, element_at, slice, size, regexp_extract, transform, when, explode, \
monotonically_increasing_id, map_from_arrays, lit, udf,collect_list, row_number, ceil, map_keys, expr, from_json
from pyspark.sql.types import ShortType, ArrayType, LongType, StringType
from pyspark.sql import types as T

directory = '/mnt/c/Users/bincl/BA-Thesis/Dataset/2gram/default/'


import gzip
import os
from os import listdir
from os.path import isfile, join
import sys

spark = SparkSession.builder.appName('3gramSQL').getOrCreate()

for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
        raw_input_df = spark \
                .read.csv(f, sep='\n',quote="").withColumnRenamed('_c0', 'Input')

        split_df = raw_input_df \
                                .select(split('Input', '\t').alias('SplitInput')) \
                                .select(element_at('SplitInput', 1).alias('Tokens'),
                                        slice('SplitInput', 2,
                size('SplitInput')).alias('Data')) \
                                .select('Tokens', 'Data') \

        df_2gram = split_df.select('Tokens', transform('Data', lambda d:
                split(d, ',')).alias('Data')) \
                                .select('Tokens', transform('Data', lambda x:
                x[0]).alias('Year'),
                                        transform('Data', lambda x:
                x[1]).cast(ArrayType(LongType())).alias('Occurrences')) \
                                .select('Tokens', map_from_arrays('Year',
                'Occurrences').alias('Data')) \
                                .select(['Tokens', 'Data'])  
                                
        df_2gram.write.parquet( "/mnt/c/Users/bincl/BA-Thesis/Dataset/2gram/parquet/" + filename,mode="overwrite")

       