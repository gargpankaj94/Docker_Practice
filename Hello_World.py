from pyspark.sql import DataFrame, SparkSession
from typing import List
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark import SparkFiles


spark = SparkSession \
       .builder \
       .appName("Hackathon") \
       .getOrCreate()

spark

#--------------------------------------------------------------------------
# Read Data
df = spark.read.format("csv").option('header', 'true').load("Parking_Violations_2013_2014.csv").limit(5000)
df.show(5)
#--------------------------------------------------------------------------
type(df)
#--------------------------------------------------------------------------
df.printSchema()
#--------------------------------------------------------------------------
ticket_df = df.toPandas()
ticket_df.head()
#--------------------------------------------------------------------------
df.describe().toPandas()
#--------------------------------------------------------------------------
import pandas as pd

d = {'Unique Entry': ticket_df.nunique(axis = 0),
        'Nan Entry': ticket_df.isnull().any()}
pd.DataFrame(data = d, index = ticket_df.columns.values)
#--------------------------------------------------------------------------

# From the above Result it make sense to drop below columns
drop_column = ['No Standing or Stopping Violation', 'Hydrant Violation',
               'Double Parking Violation', 'Latitude', 'Longitude',
               'Community Board', 'Community Council ', 'Census Tract', 'BIN', 'BBL','NTA']
ticket_df.drop(drop_column, axis = 1, inplace = True)
df = df.drop(*drop_column)

#--------------------------------------------------------------------------

# Apart from above column to determine "Violation Location" , other column do not contribute to this much so we can drop 
# them as well
drop_column = ['Plate ID','Issuer Code','Time First Observed','Vehicle Expiration Date','Date First Observed','Law Section','Sub Division','Violation Legal Code','From Hours In Effect','To Hours In Effect','Vehicle Color','Vehicle Year','Meter Number','Feet From Curb','Violation Post Code','Violation Description']
ticket_df.drop(drop_column, axis = 1, inplace = True)
df = df.drop(*drop_column)

#--------------------------------------------------------------------------

d = {'Unique Entry': ticket_df.nunique(axis = 0),
        'Nan Entry': ticket_df.isnull().any()}
pd.DataFrame(data = d, index = ticket_df.columns.values)

#--------------------------------------------------------------------------

# Apart from above column to determine "Violation Location" , other column do not contribute to this much so we can drop 
# them as well
drop_column = ["Days Parking In Effect    ", 'Street Code1', 'Street Code2', 'Street Code3']
ticket_df.drop(drop_column, axis = 1, inplace = True)
df = df.drop(*drop_column)

#--------------------------------------------------------------------------

d = {'Unique Entry': ticket_df.nunique(axis = 0),
        'Nan Entry': ticket_df.isnull().any()}
pd.DataFrame(data = d, index = ticket_df.columns.values)

#--------------------------------------------------------------------------

# Preliminary Data Visualization
# Barplot of 'Registration State'

import numpy as np
import matplotlib.pyplot as plt

x_ticks = ticket_df['Registration State'].value_counts().index
heights = ticket_df['Registration State'].value_counts()
y_pos = np.arange(len(x_ticks))
fig = plt.figure(figsize=(15,14)) 
# Create horizontal bars
plt.barh(y_pos, heights)
 
# Create names on the y-axis
plt.yticks(y_pos, x_ticks)
 
# Show graphic
plt.show()

#--------------------------------------------------------------------------

val_1 = ticket_df['Registration State'].value_counts()
val_2 = len(ticket_df)
pd.DataFrame(val_1/val_2).sort_values(by=['count'], ascending=False).head(10)

#--------------------------------------------------------------------------

# How the number of tickets given changes with each month?
import seaborn as sns
sns.set(color_codes=True)
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
​
month = []
for time_stamp in pd.to_datetime(ticket_df['Issue Date']):
    month.append(time_stamp.month)
m_count = pd.Series(month).value_counts()
​
plt.figure(figsize=(12,8))
sns.barplot(y=m_count.values, x=m_count.index, alpha=0.6)
plt.title("Number of Parking Ticket Given Each Month", fontsize=16)
plt.xlabel("Month", fontsize=16)
plt.ylabel("No. of cars", fontsize=16)
plt.show();

#--------------------------------------------------------------------------

# How many parking tickets are given for each violation code?
violation_code = ticket_df['Violation Code'].value_counts()
​
plt.figure(figsize=(16,8))
f = sns.barplot(y=violation_code.values, x=violation_code.index, alpha=0.6)
#plt.xticks(np.arange(0,101, 10.0))
f.set(xticks=np.arange(0,100, 5.0))
plt.title("Number of Parking Tickets Given for Each Violation Code", fontsize=16)
plt.xlabel("Violation Code [ X5 ]", fontsize=16)
plt.ylabel("No. of cars", fontsize=16)
plt.show();

#--------------------------------------------------------------------------

# How many parking tickets are given for each body type?
x_ticks = ticket_df['Vehicle Body Type'].value_counts().index
heights = ticket_df['Vehicle Body Type'].value_counts().values
y_pos = np.arange(len(x_ticks))
fig = plt.figure(figsize=(15,4))
f = sns.barplot(y=heights, x=y_pos, orient = 'v', alpha=0.6);
# remove labels
plt.tick_params(labelbottom='off')
plt.ylabel('No. of cars', fontsize=16);
plt.xlabel('Car models [Label turned off due to crowding. Too many types.]', fontsize=16);
plt.title('Parking ticket given for different type of car body', fontsize=16);

#--------------------------------------------------------------------------

# Top 10 car body types that get the most parking tickets are listed below :
val_1 = ticket_df['Vehicle Body Type'].value_counts()
val_2 = len(ticket_df)
df_bodytype = pd.DataFrame(val_1 / val_2).sort_values(by=['count'], ascending=False).head(10)
df_bodytype

#--------------------------------------------------------------------------

# How many parking tickets are given for each vehicle make?
val_1 = ticket_df['Vehicle Make'].value_counts()
val_2 = len(ticket_df)
df_cartype = pd.DataFrame(val_1 / val_2).sort_values(by=['count'], ascending=False).head(10)
df_cartype

#--------------------------------------------------------------------------

# Insight on violation time
timestamp = []
for time in ticket_df['Violation Time']:
    if len(str(time)) == 5:
        time = time[:2] + ':' + time[2:]
        timestamp.append(pd.to_datetime(time, errors='coerce'))
    else:
        timestamp.append(pd.NaT)
    
​
ticket_df = ticket_df.assign(Violation_Time2 = timestamp)
ticket_df.drop(['Violation Time'], axis = 1, inplace = True)
ticket_df.rename(index=str, columns={"Violation_Time2": "Violation Time"}, inplace = True)

hours = [lambda x: x.hour, ticket_df['Violation Time']]

# Getting the histogram
ticket_df.set_index('Violation Time', drop=False, inplace=True)
plt.figure(figsize=(16,8))
ticket_df['Violation Time'].groupby(pd.Grouper(freq='30Min')).count().plot(kind='bar');
plt.tick_params(labelbottom='on')
plt.ylabel('No. of cars', fontsize=16);
plt.xlabel('Day Time', fontsize=16);
plt.title('Parking ticket given at different time of the day', fontsize=16);

#--------------------------------------------------------------------------

# Parking ticket vs county
violation_county = ticket_df['Violation County'].value_counts()
​
plt.figure(figsize=(16,8))
f = sns.barplot(y=violation_county.values, x=violation_county.index, alpha=0.6)
# remove labels
plt.tick_params(labelbottom='on')
plt.ylabel('No. of cars', fontsize=16);
plt.xlabel('County', fontsize=16);
plt.title('Parking ticket given in different counties', fontsize=16);

#--------------------------------------------------------------------------

# Unregistered Vehicle?
sns.countplot(x = 'Unregistered Vehicle?', data = ticket_df)

#--------------------------------------------------------------------------

# Violation In Front Of Or Opposite
plt.figure(figsize=(16,8))
sns.countplot(x = 'Violation In Front Of Or Opposite', data = ticket_df);
  
#--------------------------------------------------------------------------

print("Row count with negative values for the following columns:")
for column in df.columns:
    print(f"'{column}': {df.filter(df[column] < 0).count()}")
  
#--------------------------------------------------------------------------

# Perform Missing Value Analysis
from pyspark.sql.functions import isnull, when, count, col
df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()

#--------------------------------------------------------------------------

drop_column = ['Violation In Front Of Or Opposite','House Number','Street Name','Intersecting Street',
               'Vehicle Make','Plate Type','Violation Precinct','Issuer Precinct','Issuer Command','Issuer Squad']
ticket_df.drop(drop_column, axis = 1, inplace = True)
df = df.drop(*drop_column)
df.printSchema()

#--------------------------------------------------------------------------

drop_column = ['Unregistered Vehicle?']
ticket_df.drop(drop_column, axis = 1, inplace = True)
df = df.drop(*drop_column)
df.printSchema()

#--------------------------------------------------------------------------

# Drop missing values
df = df.replace('null', None)\
        .dropna(how='any')

#--------------------------------------------------------------------------

# Perform Missing Value Analysis
df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()

#--------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnull, when, count, col
from pyspark.ml.stat import Correlation
from pyspark.ml.linalg import DenseMatrix, Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.sql import functions as F
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

dataset = df
dataset = dataset.withColumnRenamed('Summons Number', 'Summons_Number')
dataset = dataset.withColumnRenamed('Registration State', 'Registration_State')
dataset = dataset.withColumnRenamed('Issue Date', 'Issue_Date')
dataset = dataset.withColumnRenamed('Violation Code', 'Violation_Code')
dataset = dataset.withColumnRenamed('Vehicle Body Type', 'Vehicle_Body_Type')
dataset = dataset.withColumnRenamed('Issuing Agency', 'Issuing_Agency')
dataset = dataset.withColumnRenamed('Violation Location', 'Violation_Location')
dataset = dataset.withColumnRenamed('Violation Time', 'Violation_Time')
dataset = dataset.withColumnRenamed('Violation County', 'Violation_County')

dataset.show()

#--------------------------------------------------------------------------

# StringIndexer
from pyspark.ml.feature import StringIndexer
inputCols = ["Violation_Location" , "Summons_Number" , "Registration_State" , "Issue_Date" , "Violation_Code" , 
             "Vehicle_Body_Type" ,  "Violation_Time" , "Violation_County"]
outputCols = ["label" , "Summons_Number_i" , "Registration_State_i" , "Issue_Date_i" , "Violation_Code_i" , 
             "Vehicle_Body_Type_i" , "Violation_Time_i" , "Violation_County_i"]
​
indexers = StringIndexer(inputCols = inputCols , outputCols = outputCols)
strindexedDF = indexers.fit(dataset).transform(dataset)
strindexedDF.select('Summons_Number', 'Summons_Number_i', 'Registration_State', 'Registration_State_i', 
                     'Issue_Date', 'Issue_Date_i', 'Violation_Code' , 'Violation_Code_i',
                     'Vehicle_Body_Type', 'Vehicle_Body_Type_i', 'Violation_Time' , 'Violation_Time_i',
                     'Violation_County', 'Violation_County_i').show(5, False)

#--------------------------------------------------------------------------

# OneHotEncoderEstimator
from pyspark.ml.feature import OneHotEncoder
inputCols = ["Summons_Number_i" , "Registration_State_i" , "Issue_Date_i" , "Violation_Code_i" , 
             "Vehicle_Body_Type_i" , "Violation_Time_i" , "Violation_County_i"]
outputCols = ["Summons_Number_Vec" , "Registration_State_Vec" , "Issue_Date_Vec" , "Violation_Code_Vec" , 
             "Vehicle_Body_Type_Vec" , "Violation_Time_Vec" , "Violation_County_Vec"]
​
encoder = OneHotEncoder(inputCols = inputCols, outputCols = outputCols)
encodedDF = encoder.fit(strindexedDF).transform(strindexedDF)
encodedDF.select('Summons_Number_i', 'Summons_Number_Vec', 'Registration_State_i', 'Registration_State_Vec', 
                     'Issue_Date_i', 'Issue_Date_Vec', 'Violation_Code_i' , 'Violation_Code_Vec',
                     'Vehicle_Body_Type_i', 'Vehicle_Body_Type_Vec', 'Violation_Time_i' , 'Violation_Time_Vec',
                     'Violation_County_i', 'Violation_County_Vec').show(5, False)

encodedDF.show(5)

#--------------------------------------------------------------------------

# VectorAssembler
# Import VectorAssembler from pyspark.ml.feature package
from pyspark.ml.feature import VectorAssembler
# Create a list of all the variables that you want to create feature vectors
# These features are then further used for training model
features_col = ['Summons_Number_Vec', 'Registration_State_Vec', 'Issue_Date_Vec', 'Violation_Code_Vec',
                     'Vehicle_Body_Type_Vec', 'Violation_Time_Vec', 'Violation_County_Vec']
# Create the VectorAssembler object
assembler = VectorAssembler(inputCols= features_col, outputCol= "features")
assembledDF = assembler.transform(encodedDF)
assembledDF.select("features").show(5, False)

assembledDF.columns

#--------------------------------------------------------------------------

# VectorIndexer
# Import VectorIndexer from pyspark.ml.feature package
from pyspark.ml.feature import VectorIndexer
# Create a list of all the raw features
# VectorIndexer will automatically identify the categorical columns and index them
featurecol = ['Summons_Number_i', 'Registration_State_i', 'Issue_Date_i', 'Violation_Code_i',
                     'Vehicle_Body_Type_i', 'Violation_Time_i', 'Violation_County_i']
​
# Create the VectorAssembler object
assembler = VectorAssembler(inputCols= featurecol, outputCol= "features")
assembledDF = assembler.transform(strindexedDF)
​
# Create the VectorIndexer object. It only take feature column
vecindexer = VectorIndexer(inputCol= "features", outputCol= "indexed_features")
# Fit the vectorindexer object on the output of the vectorassembler data and transform
vecindexedDF = vecindexer.fit(assembledDF).transform(assembledDF)
vecindexedDF.select("features", "indexed_features").show(5, False)

#--------------------------------------------------------------------------

# StandardScaler
from pyspark.sql import functions as F
from pyspark.ml.linalg import Vectors, VectorUDT
​
# Define a udf that converts sparse vector into dense vector
# You cannot create your own custom function and run that against the data directly. 
# In Spark, You have to register the function first using udf function
sparseToDense = F.udf(lambda v : Vectors.dense(v), VectorUDT())
​
# We then call the function here passing the column name on which the function has to be applied
densefeatureDF = assembledDF.withColumn('features_array', sparseToDense('features'))
​
#densefeatureDF.select("features", "features_array").show(5, False)

densefeatureDF.printSchema()

#--------------------------------------------------------------------------

# Import StandardScaler from pyspark.ml.feature package
from pyspark.ml.feature import StandardScaler

# Create the StandardScaler object. It only take feature column (dense vector)
stdscaler = StandardScaler(inputCol= "features_array", outputCol= "scaledfeatures")

# Fit the StandardScaler object on the output of the dense vector data and transform
#stdscaledDF = stdscaler.fit(densefeatureDF).transform(densefeatureDF)
#stdscaledDF.select("scaledfeatures" ).show(5, False)

#--------------------------------------------------------------------------

# Train-Test Split
# We spilt the data into 70-30 set
# Training Set - 70% obesevations
# Testing Set - 30% observations
trainDF, testDF =  assembledDF.randomSplit([0.7,0.3], seed = 2020)
​
# print the count of observations in each set
print("Observations in training set = ", trainDF.count())
print("Observations in testing set = ", testDF.count())

#--------------------------------------------------------------------------

# Supervised Learning - Classification
# Logistic Regression

# import the LogisticRegression function from the pyspark.ml.classification package
from pyspark.ml.classification import LogisticRegression
​
# Build the LogisticRegression object 'lr' by setting the required parameters
lr = LogisticRegression(featuresCol="features", labelCol="label",maxIter= 10,regParam=0.3, elasticNetParam=0.8)
​
# fit the LogisticRegression object on the training data
lrmodel = lr.fit(trainDF)
​
#This LogisticRegressionModel can be used as a transformer to perform prediction on the testing data
predictonDF = lrmodel.transform(testDF)
​
predictonDF.select("label","rawPrediction", "probability", "prediction").show(10,False)

predictonDF.select("label","rawPrediction", "probability", "prediction").show(50,False)

#--------------------------------------------------------------------------

# Model Evaluation
# import BinaryClassificationEvaluator from the pyspark.ml.evaluation package
from pyspark.ml.evaluation import BinaryClassificationEvaluator
​
# Build the BinaryClassificationEvaluator object 'evaluator'
evaluator = BinaryClassificationEvaluator()
​
# Calculate the accracy and print its value
accuracy = predictonDF.filter(predictonDF.label == predictonDF.prediction).count()/float(predictonDF.count())
print("Accuracy = ", accuracy)

#--------------------------------------------------------------------------

# Create model summary object
lrmodelSummary = lrmodel.summary

# Print the following metrics one by one: 
# 1. Accuracy
# Accuracy is a model summary parameter
print("Accuracy = ", lrmodelSummary.accuracy)
# 2. Area under the ROC curve
# Area under the ROC curve is a model summary parameter
#print("Area under the ROC curve = ", lrmodelSummary.areaUnderROC)
# 3. Precision (Positive Predictive Value)
# Precision is a model summary parameter
print("Precision = ", lrmodelSummary.weightedPrecision)
# 4. Recall (True Positive Rate)
# Recall is a model summary parameter
print("Recall = ", lrmodelSummary.weightedRecall)
# 5. F1 Score (F-measure)
# F1 Score is a model summary method
print("F1 Score = ", lrmodelSummary.weightedFMeasure())

#--------------------------------------------------------------------------

