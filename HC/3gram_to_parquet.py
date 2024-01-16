from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, element_at, slice, size, regexp_extract, transform, when, explode, \
monotonically_increasing_id, map_from_arrays, lit, udf,collect_list, row_number,aggregate, ceil, map_keys, expr, from_json, sum
from pyspark.sql.types import ShortType, ArrayType, LongType, StringType
from pyspark.sql import types as T

import pandas as pd

#path = 'C:/Users/bincl/BA-Thesis/Dataset/2gram/2_20000_nopos_ab.gz'
start_date = 1800
end_date = 2000
path = '/mnt/c/Users/bincl/BA-Thesis/Dataset/3gram/default/3_20000_nopos_sample/3_20000_nopos_sample.gz'


spark = SparkSession.builder.appName('3gramSQL').getOrCreate()

raw_input_df = spark \
                .read.csv(path, sep='\n',quote="").withColumnRenamed('_c0', 'Input')

split_df = raw_input_df \
                .select(split('Input', '\t').alias('SplitInput')) \
                .select(element_at('SplitInput', 1).alias('Tokens'),
                                        slice('SplitInput', 2,
                size('SplitInput')).alias('Data')) \
                                .select('Tokens', 'Data') \

test_df_3gram = split_df.select('Tokens', transform('Data', lambda d:
                split(d, ',')).alias('Data')) \
                                .select('Tokens', transform('Data', lambda x:
                x[0]).alias('Year'),
                                        transform('Data', lambda x:
                x[1]).cast(ArrayType(LongType())).alias('Occurrences')) 
                            
df = test_df_3gram.withColumn(
    "Sum",
        aggregate("Occurrences", lit(0), lambda acc, x: (acc + x).cast("int"))
)
df_3gram =  df.select('Tokens', map_from_arrays('Year',
                'Occurrences').alias('Data') ,'Sum') \
                                .select(['Tokens', 'Data', 'Sum'])  
df_3gram = df_3gram.repartition(5)                            
                     
df_3gram.sort("Sum").write.parquet("/mnt/c/Users/bincl/BA-Thesis/Dataset/3gram/parquet/3_20000_nopos_sample/3_20000_nopos_sample.gz", mode= 'overwrite')                          
    
df= spark.read.parquet("/mnt/c/Users/bincl/BA-Thesis/Dataset/3gram/parquet/3_20000_nopos_sample/3_20000_nopos_sample.gz")
print(df.head(5))
print(df.tail(5))                  