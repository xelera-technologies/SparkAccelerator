# Following this example: https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier

CSV_PATH = "/app/data/2004_nona.csv"
APP_NAME = "Random Forest Example"
SPARK_URL = "local[*]"
RANDOM_SEED = 13579
TRAININGDATA_RATIO = 0.7
VI_MAX_CATEGORIES = 512
RF_NUM_TREES = 10
RF_MAX_DEPTH = 30
RF_MAX_BINS = 2048
REMOVED_FEATURES = ["DepTime","CRSDepTime", "ArrTime", "CRSArrTime", "FlightNum", "ActualElapsedTime", "CRSElapsedTime", "AirTime", "ArrDelay", "TaxiIn", "TaxiOut", "Cancelled", "CancellationCode", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
CATEGORICAL_FEATURES = ["UniqueCarrier", "TailNum", "Origin", "Dest"]

from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer, VectorAssembler, VectorIndexer
from pyspark.sql import functions
from pyspark.sql import SparkSession
from time import *

# Creates Spark Session
spark = SparkSession.builder.appName(APP_NAME).master(SPARK_URL).getOrCreate()

# Reads in CSV file as DataFrame
# header: The first line of files are used to name columns and are not included in data. All types are assumed to be string.
# inferSchema: Automatically infer column types. It requires one extra pass over the data.
data = spark.read.options(header = "true", inferschema = "true").csv(CSV_PATH).limit(2000000)

# Removes features that would do not add information or are unknown at prediction time (e.g. delay or cancellation code)
data = data.drop(*REMOVED_FEATURES)

# Transforms all string features into indexed numbers
# handleInvalid="keep" assigns an index to empty fields in the CancellationCode column
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index", handleInvalid="keep").fit(data) for column in CATEGORICAL_FEATURES]
pipeline = Pipeline(stages=indexers)
data = pipeline.fit(data).transform(data)

# Removes old string columns
data = data.drop(*CATEGORICAL_FEATURES)

# Creates the label, set to 1 if DepDelay >= 15, otherwise 0
data = data.withColumn("label", functions.when(functions.col("DepDelay") >= 15, 1).otherwise(0))
data = data.drop("DepDelay")

# Fill null fields with 0
data = data.fillna(0)

# Assembles all feature columns and moves them to the last column
assembler = VectorAssembler(inputCols=data.columns[0:-1], outputCol="features")
data = assembler.transform(data)

# Remove all columns but label and features
data = data.drop(*data.columns[0:-2])

# Splits the dataset into a training and testing set according to the defined ratio using the defined random seed.
splits = [TRAININGDATA_RATIO, 1.0 - TRAININGDATA_RATIO]
trainingData, testData = data.randomSplit(splits, RANDOM_SEED)

print("Number of training set rows: %d" % trainingData.count())
print("Number of test set rows: %d" % testData.count())

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > VI_MAX_CATEGORIES distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedfeatures", maxCategories=VI_MAX_CATEGORIES).fit(data)

# Train a RandomForest model.
randomForest = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedfeatures", maxBins=RF_MAX_BINS, maxDepth=RF_MAX_DEPTH, numTrees=RF_NUM_TREES)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, randomForest, labelConverter])

# Train model.  This also runs the indexers. Measures the execution time as well.
start_time = time()
model = pipeline.fit(trainingData)
end_time = time()
print("Training time:", (end_time - start_time),"s")

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))