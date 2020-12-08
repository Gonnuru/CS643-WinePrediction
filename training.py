# ########################################################################################################
#  Train.py
#
#  Sampath Gonnuru
#  sg2277@njit.edu
#  CS643 Fall 2020
#  Project 2, Model Training
#
# ########################################################################################################

#import findspark
#findspark.init()

from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col


conf = (SparkConf().setAppName("Train wine app"))
sc = SparkContext("local", conf=conf)
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

dataFrame = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', sep=';').load('s3://mywineproject/TrainingDataset.csv')
validatedataFrame = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', sep=';').load('s3://mywineproject/ValidationDataset.csv')

# dropping quality column
newDf = dataFrame.select(dataFrame.columns[:11])

outputRdd = dataFrame.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[:11])))

model = RandomForest.trainClassifier(outputRdd,numClasses=10,categoricalFeaturesInfo={}, numTrees=60, maxBins=32, maxDepth=4, seed=42)


validationOutputRdd = validatedataFrame.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[:11])))

predictions = model.predict(validationOutputRdd.map(lambda x: x.features))
labelsAndPredictions = validationOutputRdd.map(lambda lp: lp.label).zip(predictions)

metrics = MulticlassMetrics(labelsAndPredictions)

# Overall statistics
f1Score = metrics.fMeasure()
print("==== Summary Statistics ====")
print("Weighted F(1) Score = %3s" % metrics.weightedFMeasure())

print("\n\n==== Saving model ====")
#Saving model
model.save(sc, "s3://mywineproject/sg2277-wqModel")

print("Model Saved successfully")
