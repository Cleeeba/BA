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
    .config("spark.executor.memory", "10g")\
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")\
    .config("spark.default.parallelism", "96")\
        .getOrCreate()
    
    
from pyspark.sql.functions import countDistinct
import os
import seaborn as sns
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col
from matplotlib import pyplot as plt
from pyspark.sql.functions import split, element_at, map_concat,create_map, map_from_entries, when, col, lit, collect_list,expr, arrays_zip, monotonically_increasing_id, transform, arrays_zip, size, slice, collect_list, stddev_pop, avg, col , explode, col, sqrt, pow
from pyspark.sql.types import LongType, IntegerType

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

# Ngram to child Table
def load_ngramTable():
    #token_array_df = cl._CorpusLoader__contains_df
    token_array_df_cl = cl._CorpusLoader__contains_df
    array_df = cl._CorpusLoader__array_df
    token_df = cl._CorpusLoader__token_df
    data_parquet = cl._CorpusLoader__data_df
    
    #token_array_df_cl = token_array_df_cl.join(token_df, on = "TokenId")
    # Baue NgramId mit child tabell
    #token_array_df = token_array_df_cl.orderBy('NgramId', 'Position').groupBy('NgramId').agg(collect_list('TokenId').alias('Tokens'))
    
    df = array_df.withColumnRenamed('Tokens', 'Ngram').withColumn('Length', size('Ngram')).where('Length > 1')
    df = df.withColumn('LeftChildToken', slice(df.Ngram, 1, df.Length-1))
    df = df.withColumn('RightChildToken', slice(df.Ngram, 2,df.Length-1))
    
    result = (df.withColumnRenamed("Frequency","Frequency_N").alias("og")
            .join(array_df.withColumnRenamed('NgramId', 'LeftChildNgramId').withColumnRenamed('Frequency', 'Frequency_L').alias("dataL"),(col("og.LeftChildToken") == col("dataL.Tokens")))
            .join(array_df.withColumnRenamed('NgramId', 'RightChildNgramId').withColumnRenamed('Frequency', 'Frequency_R').alias("dataR"), (col("og.RightChildToken") == col("dataR.Tokens")))
            .select(col("og.NgramId").alias("NgramId"), "Frequency_N","Frequency_L", "Frequency_R"))
    return result


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


def freq_df_map(ngram_table):
    
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

def freq_df_array(ngram_table):
    
    ngram_tableN = ngram_table.withColumn("Frequency_N_new", explode("Frequency_N")).select("NgramId","Frequency_N_new")
    ngram_tableR = ngram_table.withColumn("Frequency_R_new", explode("Frequency_R")).select("NgramId","Frequency_R_new")
    ngram_tableL = ngram_table.withColumn("Frequency_L_new", explode("Frequency_L")).select("NgramId","Frequency_L_new")
    ngram_table = ngram_tableN.join(ngram_tableR, on ="NgramId").join(ngram_tableL, on ="NgramId").withColumnRenamed('Frequency_R_new', 'Frequency_R').withColumnRenamed('Frequency_N_new', 'Frequency_N').withColumnRenamed('Frequency_L_new', 'Frequency_L')
    return ngram_table

def explode_arrays(ngram_table):
    # Explodiere die Arrays an derselben Position
    ngram_table = ngram_table.select("NgramId","Frequency_R","Frequency_L", posexplode("Frequency_N").alias("pos_N", "Frequency_N_new"))
    ngram_table = ngram_table.select("NgramId","Frequency_N_new","pos_N","Frequency_L", posexplode("Frequency_R").alias("pos_R", "Frequency_R_new"))
    ngram_table = ngram_table.select("NgramId","Frequency_N_new","pos_N","pos_R","Frequency_R_new", posexplode("Frequency_L").alias("pos_L", "Frequency_L_new"))
    # Filtere die Zeilen, in denen die Positionen gleich sind
    ngram_table_filtered = ngram_table.filter((col("pos_N") == col("pos_R")) & (col("pos_N") == col("pos_L")))
    ngram_table = ngram_table_filtered.withColumnRenamed('Frequency_R_new', 'Frequency_R').withColumnRenamed('Frequency_N_new', 'Frequency_N').withColumnRenamed('Frequency_L_new', 'Frequency_L')
    # Entferne die Positionsspalten

    
    return ngram_table

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
    return full_df.select("NgramId","Aprox", "Coef_R", "Coef_L","ic")

def build_aprox_array(df, beta):
    beta = beta.withColumnRenamed("Frequency_L","Coef_L").withColumnRenamed("Frequency_R","Coef_R")
    df_1= df.groupby(group_column).agg(collect_list("Frequency_N").alias("Frequency_N"),collect_list("Frequency_L").alias("Frequency_L"),collect_list("Frequency_R").alias("Frequency_R"))

    df = beta.join(df_1, on= "NgramId")
    result_df = df.withColumn("L_multi", expr("transform(Frequency_L, x -> x * Coef_L + ic)"))
    result_df = result_df.withColumn("R_multi", expr("transform(Frequency_R, x -> x * Coef_R)"))


    full_df = result_df.withColumn("Aprox", expr("transform(L_multi, (x, i) -> x + R_multi[i])"))
    return full_df.select("NgramId","Aprox", "Coef_R", "Coef_L","ic")
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

    plt.savefig("/mnt/simhomes/binzc/png/full_small_line_plot.png")
    plt.clf()

    collect_plot = final_df.select("Sum","rmse").collect()
    pandas_df = pd.DataFrame(collect_plot, columns=["Sum", "rmse"])
    plt.scatter(x=pandas_df["Sum"], y=pandas_df["rmse"])
    plt.xscale("log")

    plt.xlabel("Sum")
    plt.ylabel("RMSE")
    plt.title("Scatter Plot mit logarithmischer Skala auf der x-Achse")
    plt.savefig("/mnt/simhomes/binzc/png/full_small_scatter_plot.png")
    plt.clf()
    sns.violinplot(x=pandas_df["rmse"], inner_kws=dict(box_width=15, whis_width=2, color=".8"))
    plt.savefig("/mnt/simhomes/binzc/png/full_violin_plot.png")
    plt.clf()
    


final = spark.read.parquet("/mnt/simhomes/binzc/parquets/full_final_df2").repartition(4*128, "NgramId")
aprox = spark.read.parquet("/mnt/simhomes/binzc/parquets/aprox_level3_table").repartition(4*128, "NgramId")
final.printSchema()
#aprox.printSchema()
data_parquet = cl._CorpusLoader__data_df
token_array_df = cl._CorpusLoader__contains_df
array_df = cl._CorpusLoader__array_df

df = array_df.withColumnRenamed('Tokens', 'Ngram').withColumn('Length', size('Ngram')).where('Length > 1')
df = df.withColumn('LeftChildToken', slice(df.Ngram, 1, df.Length-1))
df = df.withColumn('RightChildToken', slice(df.Ngram, 2,df.Length-1))
    
result = (df.withColumnRenamed("Frequency","Frequency_N").alias("og")
            .join(array_df.withColumnRenamed('NgramId', 'LeftChildNgramId').alias("dataL"),(col("og.LeftChildToken") == col("dataL.Tokens")))
            .join(array_df.withColumnRenamed('NgramId', 'RightChildNgramId').alias("dataR"), (col("og.RightChildToken") == col("dataR.Tokens")))
            .select(col("og.NgramId").alias("NgramId"), "LeftChildNgramId","RightChildNgramId"))

ngram_table = (result
                .join(final, on=("NgramId"))
                .join(aprox, on=("NgramId")).alias("og").withColumnRenamed("Aprox","lost")
                .join(aprox.alias("dataL"),(col("og.LeftChildNgramId") == col("dataL.NgramId"))).withColumnRenamed("Aprox","Frequency_L")
                .join(aprox.alias("dataR"),(col("og.RightChildNgramId") == col("dataR.NgramId"))).withColumnRenamed("Aprox","Frequency_R")
                .select(col("og.NgramId").alias("NgramId"),col("og.Frequency_N").alias("Frequency_N"),  "Frequency_L",  "Frequency_R",col("og.ic").alias("ic"),col("og.Coef_L").alias("Coef_L"),col("og.Coef_R").alias("Coef_R")))
from pyspark.sql.window import Window
from pyspark.sql.functions import first
from pyspark.sql.functions import col, explode, posexplode




freq_table              = explode_arrays(ngram_table)
df = freq_table.repartition(20*128, "NgramId")
#beta = df.groupby(group_column).applyInPandas(ols_sk,schema= schema)
beta = df.groupby(group_column).applyInPandas(ols,schema= schema)
beta.drop("Frequency_N")


from pyspark.sql.window import Window
from pyspark.sql.functions import first

beta = beta.selectExpr("NgramId", "ic","Frequency_L as Coef_L", "Frequency_R as Coef_R")
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

group_column_multi = ["NgramId", "Coef_R", "Coef_L","ic"]
aprox_group_table = aprox_table.groupBy(group_column_multi).agg(collect_list("Frequency_N").alias("Frequency_N"),collect_list("Aprox").alias("Aprox"))
params_df = aprox_group_table.select("NgramId", "Coef_R", "Coef_L","ic")
aprox_group_table.cache()
aprox_group_table = aprox_group_table.select("NgramId","Aprox","Frequency_N")
#aprox_group_table.cache()

aprox_group_table.write.parquet("/mnt/simhomes/binzc/parquets/aprox_level4_table" , mode= 'overwrite')

ZScore_df, ZScore_N_df  = Zscore_calc(aprox_group_table)

rmse_table              = RMSE(ZScore_N_df,ZScore_df)
#print(rmse_table.count())
levelTwoAprox = ZScore_df.join(ZScore_N_df, on="NgramId").join(rmse_table, on="NgramId").join(params_df, on="NgramId") 
#print(levelTwoAprox.count())

levelTwoAprox.write.parquet("/mnt/simhomes/binzc/parquets/level4_table" , mode= 'overwrite')
print(levelTwoAprox.count())






#build_graph(calc_df)