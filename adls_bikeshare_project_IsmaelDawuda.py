# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql import DataFrame


# COMMAND ----------



# Convert Python script to Jupyter notebook
!jupytext --to notebook /Workspace/Users/thomasbilintoh@yahoo.com/building_an_azure_data_lake_for_bikeshare_data_analytics/adls_bikeshare_project_IsmaelDawuda/adls_bikeshare_project_IsmaelDawuda.py


# COMMAND ----------




# COMMAND ----------

spark = SparkSession.builder.appName("bikeshare").getOrCreate()

# COMMAND ----------

[ spark.sql(f"DROP TABLE IF EXISTS {table}") for table in ['payments', 'trips', 'riders', 'stations', 'trip_dates', 'payment_dates'] ]


# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading and Writing Data to Delta Lake

# COMMAND ----------


payment_df = spark.read.format('csv').option('sep', ',').load('/FileStore/payments.csv')
trip_df = spark.read.format('csv').option('sep', ',').load('/FileStore/trips.csv')
rider_df = spark.read.format('csv').option('sep', ',').load('/FileStore/riders.csv')
station_df = spark.read.format('csv').option('sep', ',').load('/FileStore/stations.csv')


dataframes = {
    'payments': payment_df,
    'trips': trip_df,
    'riders': rider_df,
    'stations': station_df
}

for name, df in dataframes.items():
    df.write.format('delta').mode('overwrite').saveAsTable(name)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Formatting the columns to reflect the schema design

# COMMAND ----------

def format_columns(table_path: str, column_rename_dict: dict, column_type_dict: dict) -> None:
    """
    Formats columns in a Spark table by renaming and casting types.

    Args:
        table_path (str): The path to the table.
        column_rename_dict (dict): A dictionary mapping old column names to new column names.
        column_type_dict (dict): A dictionary mapping column names to their new types.

    Returns:
        None
    """
    # Read the table
    df = spark.read.table(table_path)

    # Rename columns
    df = df.select([col(c).alias(column_rename_dict.get(c, c)) for c in df.columns])

    # Cast column types
    df = df.select([col(c).cast(column_type_dict.get(c, df.schema[c].dataType)) for c in df.columns])

    # Write the transformed DataFrame back to the table
    df.write.format("delta").mode("overwrite").option("overwriteSchema", True).saveAsTable(table_path)



# Column renaming and type definitions
columns_types = {
    'payments': ({'_c0': 'payment_id', '_c1': 'date_id', '_c2': 'amount', '_c3': 'rider_id'}, {'payment_id': 'int', 'amount': 'decimal', 'date_id': 'date', 'rider_id': 'int'}),
    'trips': ({'_c0': 'trip_id', '_c1': 'rideable_type', '_c2': 'started_at', '_c3': 'ended_at', '_c4': 'start_station_id', '_c5': 'end_station_id', '_c6': 'rider_id'}, {'trip_id': 'string', 'rideable_type': 'string', 'started_at': 'timestamp', 'ended_at': 'timestamp', 'start_station_id': 'string', 'end_station_id': 'string', 'rider_id': 'int'}),
    'riders': ({'_c0': 'rider_id', '_c1': 'first', '_c2': 'last', '_c3': 'address', '_c4': 'birthday', '_c5': 'account_start_date', '_c6': 'account_end_date', '_c7': 'is_member'}, {'rider_id': 'int', 'first': 'string', 'last': 'string', 'address': 'string', 'birthday': 'date', 'account_start_date': 'date', 'account_end_date': 'date', 'is_member': 'boolean'}),
    'stations': ({'_c0': 'station_id', '_c1': 'name', '_c2': 'latitude', '_c3': 'longitude'}, {'station_id': 'string', 'name': 'string', 'latitude': 'float', 'longitude': 'float'})
}

# Apply transformations for each table
for table, (columns, types) in columns_types.items():
    format_columns(table, columns, types)



# COMMAND ----------

# MAGIC %md
# MAGIC ###Adding columns to address business outcomes

# COMMAND ----------

# Read tables
dataframes = {
    'trips': spark.read.table('trips'),
    'riders': spark.read.table('riders'),
    'payments': spark.read.table('payments')
}

# Calculate trip duration and time_id
dataframes['trips'] = dataframes['trips'].withColumn("duration", (col("ended_at") - col("started_at")).cast("long")) \
                                         .withColumn("time_id", date_trunc("hour", col("started_at")))

# Calculate age at account start
dataframes['riders'] = dataframes['riders'].withColumn("age_at_account_start", (datediff(col("account_start_date"), col("birthday")) / 365).cast("int"))

# Write updated rider data
dataframes['riders'].write.format("delta").mode("overwrite").option("overwriteSchema", True).saveAsTable('riders')

# List rider columns excluding 'rider_id'
rider_columns = [col for col in dataframes['riders'].columns if col != 'rider_id']

# Join trip and rider data, calculate age at ride time
dataframes['trips'] = dataframes['trips'].join(dataframes['riders'].select('rider_id', 'birthday'), on='rider_id', how='inner') \
                                         .withColumn("age_at_ride_time", (datediff(to_date(col("started_at")), col("birthday")) / 365).cast("int")) \
                                         .select('trip_id', 'duration', 'rideable_type', 'age_at_ride_time', 'started_at', 'ended_at', 'start_station_id', 'end_station_id', 'time_id', 'rider_id')

# Write updated trip data
dataframes['trips'].write.format("delta").mode("overwrite").option("overwriteSchema", True).saveAsTable('trips')

# Write payment data
dataframes['payments'].select('payment_id', 'amount', 'date_id', 'rider_id').write.format("delta").mode("overwrite").option("overwriteSchema", True).saveAsTable('payments')


# COMMAND ----------

# MAGIC %md
# MAGIC ### Date Dimensions
# MAGIC Separate date dimension tables will be created for payment and trip data due to differences in their time granularity:
# MAGIC
# MAGIC The trip date dimension captures time-of-day info (morning, afternoon, evening, night) at an hourly level. The payment date dimension focuses on spending trends by month, quarter, and year at a daily level.

# COMMAND ----------

# Read and cache tables
payment_df, trip_df = (spark.read.table('payments').cache(), spark.read.table('trips').cache())

# Get min and max dates for payment and trip
payment_min_date, payment_max_date = payment_df.select(min('date_id'), max('date_id')).first()
trip_min_date, trip_max_date = trip_df.select(min('time_id'), max('time_id')).first()

# Log date ranges
print(f"Trip Dates: {trip_min_date} to {trip_max_date}")
print(f"Payment Dates: {payment_min_date} to {payment_max_date}")

# Create date and time sequences
sequences = [
    spark.sql(f"SELECT explode(sequence(to_date('{payment_min_date}'), to_date('{payment_max_date}'), INTERVAL 1 DAY)) AS date").createOrReplaceTempView('payment_dates_view'),
    spark.sql(f"SELECT explode(sequence(to_timestamp('{trip_min_date}'), to_timestamp('{trip_max_date}'), INTERVAL 1 HOUR)) AS time").createOrReplaceTempView('trip_dates_view')
]


# COMMAND ----------

# MAGIC %sql SELECT * FROM trip_dates_view LIMIT 20

# COMMAND ----------

trip_dates_query = f"""
SELECT
    time AS time_id,
    dayofweek(time) AS day_of_week,
    CASE 
        WHEN hour(time) BETWEEN 5 AND 11 THEN 'morning'
        WHEN hour(time) BETWEEN 12 AND 16 THEN 'afternoon'
        WHEN hour(time) BETWEEN 17 AND 21 THEN 'evening'
        ELSE 'night'
    END AS time_of_day
FROM trip_dates_view
ORDER BY time
"""

trip_dates = spark.sql(trip_dates_query)
trip_dates.write.format('delta').mode('overwrite').saveAsTable('trip_dates')


# COMMAND ----------

# Define the SQL query  for payment dates
payment_dates_query = f"""
SELECT
    date AS date_id,
    month(date) AS month,
    quarter(date) AS quarter,
    year(date) AS year
FROM payment_dates_view
ORDER BY date
"""

payment_dates = spark.sql(payment_dates_query)
payment_dates.write.format('delta').mode('overwrite').saveAsTable('payment_dates')


# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Questions to Address
# MAGIC
# MAGIC - Analyze how much time is spent per ride
# MAGIC   * Based on date and time factors such as day of week and time of day
# MAGIC   * Based on which station is the starting and / or ending station
# MAGIC   * Based on age of the rider at time of the ride
# MAGIC   * Based on whether the rider is a member or a casual rider
# MAGIC - Analyze how much money is spent
# MAGIC   * Per month, quarter, year
# MAGIC   * Per member, based on the age of the rider at account start
# MAGIC - EXTRA CREDIT - Analyze how much money is spent per member
# MAGIC   * Based on how many rides the rider averages per month
# MAGIC   * Based on how many minutes the rider spends on a bike per month

# COMMAND ----------

# Load the fact and dimension tables
tables = ['payments', 'trips', 'riders', 'stations', 'trip_dates', 'payment_dates']
payment_df, trip_df, rider_df, station_df, trip_date_df, payment_date_df = [spark.read.table(table) for table in tables]


# COMMAND ----------

# MAGIC %md
# MAGIC ## Trip Table Queries

# COMMAND ----------

def analyze_trip_data(df: DataFrame, group_col: str, agg_func: Column, alias: str) -> None:
    """
    Analyzes trip data by joining with the trip date DataFrame and applying an aggregation function.

    Args:
        df (DataFrame): The DataFrame containing trip data.
        group_col (str): The column to group by.
        agg_func (Column): The aggregation function to apply (e.g., avg, sum).
        alias (str): The alias for the aggregated column.

    Returns:
        None
    """
    df.join(trip_date_df, 'time_id')\
        .groupBy(group_col)\
        .agg(agg_func('duration').alias(alias))\
        .orderBy(alias, ascending=False)\
        .show()


# Analyze how much time is spent per ride on average based on day of week
analyze_trip_data(trip_df, 'day_of_week', avg, 'duration_in_seconds_avg')

# Analyze how much time is spent per ride in total based on day of week
analyze_trip_data(trip_df, 'day_of_week', sum, 'duration_in_seconds_sum')

# Analyze how much time is spent per ride on average based on time of day
analyze_trip_data(trip_df, 'time_of_day', avg, 'duration_in_seconds_avg')

# Analyze how much time is spent per ride in total based on time of day
analyze_trip_data(trip_df, 'time_of_day', sum, 'duration_in_seconds_sum')


# COMMAND ----------


def analyze_duration(df: DataFrame, group_col: str, agg_func: Column, alias: str) -> None:
    """
    Analyzes duration data by grouping and applying an aggregation function.

    Args:
        df (DataFrame): The DataFrame containing duration data.
        group_col (str): The column to group by.
        agg_func (Column): The aggregation function to apply (e.g., avg, sum).
        alias (str): The alias for the aggregated column.

    Returns:
        None
    """
    df.groupBy(group_col)\
      .agg(agg_func('duration').alias(alias))\
      .orderBy(alias, ascending=False)\
      .show()

# Avg and total duration per ride by start station
analyze_duration(trip_df, 'start_station_id', avg, 'duration_in_seconds_avg')
analyze_duration(trip_df, 'start_station_id', sum, 'duration_in_seconds_sum')

# Avg and total duration per ride by end station
analyze_duration(trip_df, 'end_station_id', avg, 'duration_in_seconds_avg')
analyze_duration(trip_df, 'end_station_id', sum, 'duration_in_seconds_sum')


# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql.functions import Column

def analyze_duration_by_age(df: DataFrame, group_col: str, agg_func: Column, alias: str) -> None:
    """
    Analyzes duration data by joining with the rider DataFrame, grouping by the specified column, and applying an aggregation function.

    Args:
        df (DataFrame): The DataFrame containing duration data.
        group_col (str): The column to group by.
        agg_func (Column): The aggregation function to apply (e.g., avg, sum).
        alias (str): The alias for the aggregated column.

    Returns:
        None
    """
    df.join(rider_df, df.rider_id == rider_df.rider_id)\
      .groupBy(group_col)\
      .agg(agg_func('duration').alias(alias))\
      .orderBy(alias, ascending=False)\
      .show()


# Avg and total duration by age at account start
analyze_duration_by_age(trip_df, 'age_at_account_start', avg, 'duration_in_seconds_avg')
analyze_duration_by_age(trip_df, 'age_at_account_start', sum, 'duration_in_seconds_sum')


# COMMAND ----------

def analyze_duration_by_membership(df: DataFrame, group_col: str, agg_func: Column, alias: str) -> None:
    """
    Analyzes duration data by joining with the rider DataFrame, grouping by membership status, and applying an aggregation function.

    Args:
        df (DataFrame): The DataFrame containing duration data.
        group_col (str): The column to group by.
        agg_func (Column): The aggregation function to apply (e.g., avg, sum).
        alias (str): The alias for the aggregated column.

    Returns:
        None
    """
    df.join(rider_df, 'rider_id')\
      .groupBy(group_col)\
      .agg(agg_func('duration').alias(alias))\
      .orderBy(alias, ascending=False)\
      .show()


# Avg and total duration by rider membership
analyze_duration_by_membership(trip_df, 'is_member', avg, 'duration_in_seconds_avg')
analyze_duration_by_membership(trip_df, 'is_member', sum, 'duration_in_seconds_sum')


# COMMAND ----------

# MAGIC %md
# MAGIC ## Payment Table Queries For Analyzing Payment Data

# COMMAND ----------

def analyze_payment_data(df: DataFrame, group_col: str, agg_funcs: list, aliases: list):
    for agg_func, alias in zip(agg_funcs, aliases):
        df.join(payment_date_df, 'date_id')\
          .groupBy(group_col)\
          .agg(agg_func('amount').alias(alias))\
          .orderBy(alias, ascending=False)\
          .show()

# Aggregation functions and their aliases
agg_funcs = [sum, avg]
aliases = ['amount_sum', 'amount_avg']

# Analyze spending by different time periods
for group_col in ['month', 'quarter', 'year']:
    analyze_payment_data(payment_df, group_col, agg_funcs, aliases)


# COMMAND ----------



def analyze_member_payment_data(df: DataFrame, group_col: str, agg_func: Column, alias: str) -> None:
    """
    Analyzes payment data for members by joining with the rider DataFrame, 
    grouping by the specified column, and applying an aggregation function.

    Args:
        df (DataFrame): The DataFrame containing payment data.
        group_col (str): The column to group by.
        agg_func (Column): The aggregation function to apply (e.g., avg, sum).
        alias (str): The alias for the aggregated column.

    Returns:
        None
    """
    df.join(rider_df, 'rider_id')\
      .where(rider_df.is_member == True)\
      .groupBy(group_col)\
      .agg(agg_func('amount').alias(alias))\
      .orderBy(alias, ascending=False)\
      .show()

# Agg functions and their aliases
agg_funcs = [avg, sum]
aliases = ['amount_avg', 'amount_sum']

# Analyze spending by members by age at account start
for agg_func, alias in zip(agg_funcs, aliases):
    analyze_member_payment_data(payment_df, 'age_at_account_start', agg_func, alias)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Extra Credit 

# COMMAND ----------

# Avg spending per member by monthly ride count
trip_df.join(payment_df, 'rider_id')\
    .select('rider_id', 'time_id', 'amount', 'trip_id')\
    .join(rider_df.where(rider_df.is_member == True), 'rider_id')\
    .withColumn('month', month('time_id'))\
    .groupby('rider_id', 'month')\
    .agg(avg('amount').alias('avg_amount'), count('trip_id').alias('num_rides'))\
    .orderBy('num_rides', ascending=False)\
    .show()


# COMMAND ----------

# Avg spending per member by monthly bike usage
trip_df.join(rider_df, 'rider_id')\
    .join(payment_df, 'rider_id')\
    .filter(rider_df.is_member)\
    .withColumn('month', month('time_id'))\
    .withColumn('minutes', (trip_df.duration / 60).cast('int'))\
    .groupBy('rider_id', 'minutes', 'month')\
    .agg(
        avg('amount').alias('avg_amount'),
        avg('duration').alias('avg_duration')
    )\
    .orderBy('avg_duration', ascending=False)\
    .show()

# Investigate extended usage of a specific rider
trip_df.filter(trip_df.rider_id == 1088)\
    .select('rider_id', 'started_at', 'ended_at', 'duration')\
    .orderBy('duration', ascending=False)\
    .show()

