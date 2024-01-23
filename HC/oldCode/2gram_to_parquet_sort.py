from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, element_at, slice, size, regexp_extract, transform, when, explode, \
monotonically_increasing_id, map_from_arrays, lit, udf,collect_list, row_number,aggregate, ceil, map_keys, expr, from_json, sum, 
from pyspark.sql.types import ShortType, ArrayType, LongType, StringType
from pyspark.sql import types as T

from pyspark.conf import SparkConf
import pandas as pd
import os

#path = 'C:/Users/bincl/BA-Thesis/Dataset/2gram/2_20000_nopos_ab.gz'
start_date = 1800
end_date = 2000
directory = '/mnt/c/Users/bincl/BA-Thesis/Dataset/2gram/default'

conf= SparkConf().setAll([('spark.executor.memory', '16g'), ('spark.executor.cores', '4'), ('spark.cores.max', '4'), ('spark.driver.memory','16g')])

spark = SparkSession.builder.config(conf=conf).appName('3gramSQL').getOrCreate()

filelist = [] 
for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        filelist.append(f)
        
raw_input_df = spark \
                .read.csv(filelist, sep='\n',quote="").withColumnRenamed('_c0', 'Input')

split_df = raw_input_df \
                                .select(split('Input', '\t').alias('SplitInput')) \
                                .select(element_at('SplitInput', 1).alias('Tokens'),
                                        slice('SplitInput', 2,
                size('SplitInput')).alias('Data')) \
                                .select('Tokens', 'Data') \
                                    
test_df_2gram = split_df.select('Tokens', transform('Data', lambda d:
                split(d, ',')).alias('Data')) \
                                .select('Tokens', transform('Data', lambda x:
                x[0]).alias('Year'),
                                        transform('Data', lambda x:
                x[1]).cast(ArrayType(LongType())).alias('Occurrences')) 
                            
df = test_df_2gram.withColumn(
            "Sum",
                aggregate("Occurrences", lit(0), lambda acc, x: (acc + x).cast("int"))
        )
df_2gram =  df.select('Tokens', map_from_arrays('Year',
                        'Occurrences').alias('Data') ,'Sum') \
                                        .select(['Tokens', 'Data', 'Sum'])  
df_2gram = df_2gram.repartition(8)                            
     
                                
df_2gram.orderBy("Sum", ascending = False).write.parquet("/mnt/c/Users/bincl/BA-Thesis/Dataset/2gram/parquet/orderBy/2gram_order" , mode= 'overwrite')       
                           
df= spark.read.parquet("/mnt/c/Users/bincl/BA-Thesis/Dataset/2gram/parquet/orderBy/2gram_order")
print(df.head(5))
