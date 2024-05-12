from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pymongo import MongoClient
from annoy import AnnoyIndex
import numpy as np

# Connect to MongoDB and retrieve the data
mongo_client = MongoClient('mongodb://localhost:27017/')
mongo_db = mongo_client['music_database']
mongo_collection = mongo_db['audio_features']

data_from_mongo = list(mongo_collection.find())

# Prepare the data for training
spark_session = SparkSession.builder \
    .appName("MusicRecommendationSystem") \
    .getOrCreate()

# Convert MongoDB data to DataFrame
mongo_rows = [(doc['_id'], doc['file_name'], doc['file_path'], np.array(doc['mfcc'])) for doc in data_from_mongo]
df = spark_session.createDataFrame(mongo_rows, ["id", "file_name", "file_path", "mfcc"])

# Flatten the MFCC arrays into a single feature vector
mfcc_dimension = len(mongo_rows[0][3])
vector_assembler = VectorAssembler(inputCols=["mfcc"], outputCol="features")
df = vector_assembler.transform(df)

# Train a recommendation model
als_model = ALS(
    maxIter=10,
    regParam=0.01,
    userCol="id",
    itemCol="id",
    ratingCol="features",  # Using features as ratings for ALS
    coldStartStrategy="drop"
)

trained_model = als_model.fit(df)

# Build an ANN index for item similarity
item_embeddings = trained_model.itemFactors.rdd.map(lambda x: (x.id, x.features)).collect()
annoy_idx = AnnoyIndex(mfcc_dimension, 'euclidean')  # Assuming Euclidean distance

for item_id, embedding in item_embeddings:
    annoy_idx.add_item(item_id, embedding)

annoy_idx.build(10)  # Build the index with 10 trees for faster querying

# Function to find similar items using ANN
def find_similar_items(item_id, top_k=5):
    similar_item_ids = annoy_idx.get_nns_by_item(item_id, top_k)
    return similar_item_ids

similar_items = find_similar_items(item_id='your_item_id', top_k=5)
print(similar_items)

# Perform hyperparameter tuning using cross-validation
param_grid = ParamGridBuilder() \
    .addGrid(als_model.rank, [10, 20, 30]) \
    .addGrid(als_model.maxIter, [10, 20]) \
    .addGrid(als_model.regParam, [0.01, 0.1, 0.5]) \
    .build()

cross_validator = CrossValidator(
    estimator=als_model,
    estimatorParamMaps=param_grid,
    evaluator=RegressionEvaluator(metricName="rmse"),
    numFolds=5
)

# Fit ALS model to the DataFrame
cv_model = cross_validator.fit(df)

# Get the best ALS model from cross-validation
best_als_model = cv_model.bestModel

# Print the best parameters found during cross-validation
print("Best rank:", best_als_model.rank)
print("Best maxIter:", best_als_model._java_obj.parent().getMaxIter())
print("Best regParam:", best_als_model._java_obj.parent().getRegParam())

# Stop Spark session
spark_session.stop()
