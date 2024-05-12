import librosa
import pymongo
import numpy as np
import os
from annoy import AnnoyIndex as AI

# Set the directory containing subdirectories with audio files
root_directory = '/path/to/audio_files'

# Connect to MongoDB database
mongo_client = pymongo.MongoClient('mongodb://localhost:27017')
db = mongo_client['audio_db']
collection = db['audio_features']

# Create an Annoy index with the same number of features as MFCC array
annoy_idx = AI(len(next(iter(collection.find()))['mfcc']), 'angular')

# Iterate through database documents and add MFCC features to Annoy index
for idx, doc in enumerate(collection.find()):
    mfcc_features = doc['mfcc']
    annoy_idx.add_item(idx, mfcc_features)

# Build the Annoy index with 10 trees
annoy_idx.build(10)

# Define a function to query Annoy index for k nearest neighbors
def find_nearest_neighbors(file_name, k=5):
    document = collection.find_one({'file_name': file_name})
    if document:
        mfcc_features = document['mfcc']
        nearest_neighbors = annoy_idx.get_nns_by_vector(mfcc_features, k, search_k=-1, include_distances=True)
        return [(collection.find_one({'_id': idx})['file_name'], distance) for idx, distance in zip(nearest_neighbors[0], nearest_neighbors[1])]
    else:
        return []

# Example: Find 5 nearest neighbors for a given file
nearest_neighbors = find_nearest_neighbors('example.mp3')
print(nearest_neighbors)

# Close MongoDB client connection
mongo_client.close()
