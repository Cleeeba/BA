from itertools import chain
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

# Ngram to child Table
def load_ngramTable():
    #token_array_df = cl._CorpusLoader__contains_df
    token_array_df = cl._CorpusLoader__contains_df.limit(70000).repartition(7 * 128, col(group_column))
    data_parquet = cl._CorpusLoader__data_df.repartition(7 * 128, col(group_column))
    # Baue NgramId mit child tabelle
    token_array_df = token_array_df.orderBy('NgramId', 'Position').groupBy('NgramId').agg(collect_list('TokenId').alias('Tokens'))
    df = token_array_df.withColumnRenamed('Tokens', 'Ngram').withColumn('Length', size('Ngram')).where('Length > 1')
    df = df.withColumn('LeftChildTokenIds', slice(df.Ngram, 1, df.Length - 1))
    df = df.withColumn('RightChildTokenIds', slice(df.Ngram, 2,df.Length - 1)).coalesce(6*128)
    
    result = (df
            .join(token_array_df.withColumnRenamed('NgramId', 'LeftChildNgramId'), on= df.LeftChildTokenIds == token_array_df.Tokens).withColumnRenamed('Tokens', 'LCTokens')
            .join(token_array_df.withColumnRenamed('NgramId', 'RightChildNgramId'), on= df.RightChildTokenIds == token_array_df.Tokens).withColumnRenamed('Tokens', 'RCTokens')
            .select("NgramId", "LeftChildNgramId", "RightChildNgramId"))
    # Frequency Table without 0

    ngram_table = (result
                .join(data_parquet, on=("NgramId")).withColumnRenamed("Frequency","Frequency_N").alias("og")
                .join(data_parquet.alias("dataL"),(col("og.LeftChildNgramId") == col("dataL.NgramId"))).withColumnRenamed("Frequency","Frequency_L")
                .join(data_parquet.alias("dataR"),(col("og.RightChildNgramId") == col("dataR.NgramId"))).withColumnRenamed("Frequency","Frequency_R")
                .select(col("og.NgramId").alias("NgramId"),"Frequency_N", "Frequency_L",col("dataL.NgramId").alias("NgramIdL"), "Frequency_R",col("dataR.NgramId").alias("NgramIdR")))
    return ngram_table


def explode_and_filter(df, frequency_col, dict_years):
    
    new_col_name = f"new_{frequency_col}"
    
    result_df = df.withColumn(
        new_col_name,
        map_concat(
            create_map(*dict_years),
            frequency_col
        )
    ).select("NgramId", explode(new_col_name).alias("Year", "value")).filter((col("Year") >= 1800) & (col("Year") < 2001))

    return result_df
# Frequency Table with 0 entries
#sehr slow
def new_freq_df(ngram_table):
    
    df_N = explode_and_filter(ngram_table, "Frequency_N", dict_years)
    df_L = explode_and_filter(ngram_table, "Frequency_L", dict_years)
    df_R = explode_and_filter(ngram_table, "Frequency_R", dict_years)

    df = df_N.withColumnRenamed("value","Frequency_N").join(df_L,["NgramId","Year"])\
        .withColumnRenamed("value","Frequency_L").join(df_R,["NgramId","Year"])\
        .withColumnRenamed("value","Frequency_R").orderBy(["NgramId","Year"])
    return df


def freq_df(ngram_table):
    
    year_df = spark.createDataFrame(list(range(1800, 2001)), schema=IntegerType())
    df = ngram_table.crossJoin(year_df).withColumnRenamed('value', 'Year')
    
    df = df.withColumn("TrueFreq", when(col("Frequency_N").getItem(df.Year).isNotNull(),col("Frequency_N").getItem(df.Year)).otherwise(lit(0)))
    df = df.select("NgramId","TrueFreq", "Frequency_L", "Frequency_R",'Year').withColumnRenamed('TrueFreq', 'Frequency_N')

    df = df.withColumn("TrueFreq", when(col("Frequency_L").getItem(df.Year).isNotNull(),col("Frequency_L").getItem(df.Year)).otherwise(lit(0)))
    df = df.select("NgramId","Frequency_N", "TrueFreq", "Frequency_R" ,'Year').withColumnRenamed('TrueFreq', 'Frequency_L')
    df = df.withColumn("TrueFreq", when(col("Frequency_R").getItem(df.Year).isNotNull(),col("Frequency_R").getItem(df.Year)).otherwise(lit(0)))
    df = df.select("NgramId","Frequency_N", "Frequency_L", "TrueFreq", 'Year').withColumnRenamed('TrueFreq', 'Frequency_R')
    df_1 = df.select("NgramId","Frequency_N","Frequency_L","Frequency_R" )
    return df_1

def ols(pdf):
    group_key = pdf[group_column].iloc[0]
    y = pdf[y_column]
    X = pdf[x_columns]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return pd.DataFrame([[group_key] + [model.params[i] for i in x_columns] + [model.params['const']]], columns=[group_column] + x_columns + [intercept])

def build_aprox(df, beta):
    beta = beta.withColumnRenamed("Frequency_L","Coef_L").withColumnRenamed("Frequency_R","Coef_R")
    df_1= df.groupby(group_column).agg(collect_list("Frequency_N").alias("Frequency_N"),collect_list("Frequency_L").alias("Frequency_L"),collect_list("Frequency_R").alias("Frequency_R"))

    df = beta.join(df_1, on= "NgramId")
    result_df = df.withColumn("L_multi", expr("transform(Frequency_L, x -> x * Coef_L + ic)"))
    result_df = result_df.withColumn("R_multi", expr("transform(Frequency_R, x -> x * Coef_R)"))


    full_df = result_df.withColumn("Aprox", expr("transform(L_multi, (x, i) -> x + R_multi[i])"))
    return full_df.select("NgramId","Aprox","L_multi","R_multi")

def Zscore_calc(df):
    result_df = df.select("NgramId","Aprox", explode("Aprox").alias("exploded_aprox"))

    stats = (result_df.groupBy("NgramId")
        .agg(
            stddev_pop("exploded_aprox").alias("sd"), 
            avg("exploded_aprox").alias("avg")))


    result_df= result_df.join(stats, ["NgramId"]).select("NgramId",((result_df.exploded_aprox - stats.avg) / stats.sd).alias("ZScore"))
    ZScore_df= result_df.groupBy("NgramId").agg(collect_list("ZScore").alias("ZScoreArray"))
    #ZScore_df.show()
    # Falls fehler dann hier approx Zscore sind zu nah aneinander--------------------- Brodcast entfernt muss getestet werden ob klappt
    ZScore_N_df = df.select("NgramId","Frequency_N",explode("Frequency_N").alias("exploded_Frequency_N"))
    stats = (ZScore_N_df.groupBy("NgramId")
    .agg(
        stddev_pop("exploded_Frequency_N").alias("sd"), 
        avg("exploded_Frequency_N").alias("avg")))

    ZScore_N_df= ZScore_N_df.join(stats, ["NgramId"]).select("NgramId","Frequency_N",((ZScore_N_df.exploded_Frequency_N - stats.avg) / stats.sd).alias("ZScore_N"))
    ZScore_N_df= ZScore_N_df.groupBy("NgramId").agg(collect_list("ZScore_N").alias("ZScore_N_Array"))
    
    return ZScore_df,ZScore_N_df

#sehr langsam
def old_Sum_calc(df):
    sum_df = df.select(
    "NgramId",'Frequency_N',
    expr('AGGREGATE(Frequency_N, 0, (acc, x) -> CAST(acc AS INT) + CAST(x AS INT))').alias('Sum')
    )
    return sum_df

def Sum_no_zero(df):
    sum_df = df.select(
    "NgramId",explode('Frequency_N').alias("key", "value")).groupBy("NgramId").sum("value")\
        .select("NgramId","sum(value)").withColumnRenamed("sum(value)", "Sum")
    return sum_df

def RMSE(ZScore_N_df,ZScore_df):

    rmse_df = ZScore_N_df.join(ZScore_df, on="NgramId")

    rmse_df = rmse_df.withColumn("zip", arrays_zip("ZScoreArray", "ZScore_N_Array"))\
        .withColumn("zip", explode("zip"))\
        .select("NgramId", col("zip.ZScoreArray").alias("ZScoreArray"), col("zip.ZScore_N_Array").alias("ZScore_N_Array"))

    rmse_stats = (rmse_df.groupBy("NgramId")
    .agg(
        sqrt(avg(pow(col("ZScoreArray") - col("ZScore_N_Array"), 2))).alias("rmse")))
    return rmse_stats

def build_graph(final_df):
    line_plot = final_df.select("ZScore_N_Array", "ZScoreArray").first()
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
    
    start_rmse = time.time()
    collect_plot = final_df.select("Sum","rmse").toPandas()
    print(time.time()- start_rmse)
    print("pandas time muss < 15 min")
    
    start_rmse = time.time()
    collect_plot = final_df.select("Sum","rmse").collect()
    print(time.time()- start_rmse)
    print("collect time muss < 15 min")
    pandas_df = pd.DataFrame(collect_plot, columns=["Sum", "rmse"])
    plt.scatter(x=pandas_df["Sum"], y=pandas_df["rmse"])
    plt.xscale("log")

    plt.xlabel("Sum")
    plt.ylabel("RMSE")
    plt.title("Scatter Plot mit logarithmischer Skala auf der x-Achse")
    plt.savefig("/mnt/simhomes/binzc/png/small_scatter_plot.png")
    plt.clf()
    sns.violinplot(x=pandas_df["rmse"], inner_kws=dict(box_width=15, whis_width=2, color=".8"))
    plt.savefig("/mnt/simhomes/binzc/png/small_violin_plot.png")
    plt.clf()
    

#ngram_table.write.parquet("/mnt/c/Users/bincl/BA-Thesis/Dataset/parquets_corpus/parquets/freq", mode= 'overwrite'))
#ngram_table = spark.read.parquet("/mnt/c/Users/bincl/BA-Thesis/Dataset/parquets_corpus/parquets/freq").limit(10)

import time
start1 = time.time()
#ngram_table = spark.read.parquet("/mnt/simhomes/binzc/data_transfer/freq_small")
ngram_table             = load_ngramTable()
ngram_table.persist()
ngram_table.show()
print(time.time() - start1)
print("readin time")
start = time.time()

start_sum_0 = time.time()
sum_table = Sum_no_zero(ngram_table)
sum_table.show()
print(time.time()- start_sum_0)
print("sum_no_zero zeit")

freq_table              = freq_df(ngram_table)
freq_table.show()

print(time.time() - start)
print("freq zeit")
#df = freq_table

#df = freq_table.coalesce(4*128)
df = freq_table.repartition(4* 128, col(group_column))

start_aprox = time.time()
beta = df.groupby(group_column).applyInPandas(ols,schema= schema)
beta.drop("Frequency_N")
beta.show()
print(time.time()- start_aprox)
print("mlr zeit")


from pyspark.sql.window import Window
from pyspark.sql.functions import first

beta = beta.selectExpr("NgramId", "ic","Frequency_L as Coef_L", "Frequency_R as Coef_R")
start_build = time.time()
df = beta.join(df, on = "NgramId")
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
aprox_table = aprox_table.groupby(group_column).agg(collect_list("Frequency_N").alias("Frequency_N"),collect_list("Aprox").alias("Aprox"))

aprox_table.show()
aprox_table = aprox_table.persist()

print(time.time()- start_build)
print("build time")



start_Z = time.time()
ZScore_df, ZScore_N_df  = Zscore_calc(aprox_table)
ZScore_df.show()
ZScore_N_df.show()
print(time.time()- start_Z)
print("z time")

start_rmse = time.time()
rmse_table              = RMSE(ZScore_N_df,ZScore_df)
rmse_table.show()
print(time.time()- start_rmse)
print("rmse time")

final  = sum_table.join(rmse_table, on="NgramId").join(ZScore_N_df,on="NgramId").join(ZScore_df,on="NgramId")
final= final.select("NgramId","Sum","rmse","ZScore_N_Array","ZScoreArray").persist()

#final.coalesce(2*128).sortWithinPartitions("Sum").write.parquet("/mnt/simhomes/binzc/data_transfer/small_final_df" , mode= 'overwrite')
build_graph(final)

while(True):
    pass

