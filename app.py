from flask import Flask, render_template, request, send_from_directory
import os
import platform
import librosa
import numpy as np
import pandas as pd
import annoy
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans

app = Flask(__name__, template_folder="templates", static_folder="static")

if platform.system() == "Windows":
    app.config["UPLOAD_FOLDER"] = r"static\files"
else:
    app.config["UPLOAD_FOLDER"] = r"static/files"

def load_annoy_index():
    global annoy_index
    annoy_index = annoy.AnnoyIndex(40, "angular")
    annoy_index.load("music.ann")

def load_features():
    global feature_dataframe, feature_array
    feature_dataframe = pd.read_pickle("features.pkl")
    feature_array = np.array(feature_dataframe.Feature.tolist())

def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

def get_nearest_neighbours(new_audio, annoy_index, n_neighbours):
    new_audio_mfcc = extract_features(new_audio)
    nearest_neighbours = annoy_index.get_nns_by_vector(new_audio_mfcc, n_neighbours)
    return nearest_neighbours

def compare_vectors(new_mfcc, mfcc, nearest_neighbours):
    distances = [cosine(new_mfcc, mfcc[neighbour]) for neighbour in nearest_neighbours]
    return distances

def get_best_match(new_audio, mfcc, annoy_index, n_neighbours):
    nearest_neighbours = get_nearest_neighbours(new_audio, annoy_index, n_neighbours)
    distances = compare_vectors(extract_features(new_audio), mfcc, nearest_neighbours)
    best_match = nearest_neighbours[np.argmin(distances)]
    return best_match

def get_worst_match(new_audio, mfcc, annoy_index, n_neighbours):
    nearest_neighbours = get_nearest_neighbours(new_audio, annoy_index, n_neighbours)
    distances = compare_vectors(extract_features(new_audio), mfcc, nearest_neighbours)
    worst_match = nearest_neighbours[np.argmax(distances)]
    return worst_match

def audio_segmentation(new_audio, annoy_index, n_neighbours, n_clusters):
    neighbours = get_nearest_neighbours(new_audio, annoy_index, n_neighbours)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, algorithm="elkan").fit(feature_array[neighbours])
    cluster_labels = kmeans.labels_
    cluster_dataframe = pd.DataFrame({"Cluster": cluster_labels, "Song": feature_dataframe.Label[neighbours]})
    cluster_dataframe = cluster_dataframe.sort_values(by=["Cluster"])
    cluster_dataframe = cluster_dataframe.reset_index(drop=True)
    cluster_dataframe.to_csv("pied_piper_download.csv", index=False)

load_annoy_index()
load_features()

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(app.root_path, "static"), "musical_icon.png", mimetype="image/vnd.microsoft.icon")

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        file_name = file.filename
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file_name)
        file.save(file_path)
        best_match = get_best_match(file_path, feature_array, annoy_index, 100)
        worst_match = get_worst_match(file_path, feature_array, annoy_index, 100)
        song_one_path = feature_dataframe.Label[best_match]
        song_two_path = feature_dataframe.Label[worst_match]
        audio_segmentation(file_path, annoy_index, 100, 10)
    return render_template("predict.html", song_one_name=song_one_path, song_one_path=song_one_path, song_two_name=song_two_path, song_two_path=song_two_path)

if __name__ == "__main__":
    app.run(debug=True)
