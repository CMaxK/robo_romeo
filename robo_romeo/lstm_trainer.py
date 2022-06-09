import numpy as np
import joblib
import pickle
from google.cloud import storage
from lstm_model import LstmModel

### GCP configuration - - - - - - - - - - - - - - - - - - -
GCP_PROJECT = 'robo_romeo'
BUCKET_NAME = 'wagon-data-900-robo_romeo'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -
BUCKET_X1_TRAIN_DATA = 'extracted_features/extract_features_6k.pkl'
BUCKET_X2_TRAIN_DATA = 'processed_captions/cap4'

##### Model - - - - - - - - - - - - - - - - - - - - - - - -
MODEL_NAME = 'lstm_trainer'
MODEL_VERSION = 'v1'
STORAGE_LOCATION = 'models/lstm_model/'
JOBLIB_MODEL = 'trained_lstm_model'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

def get_data():
    features_file = 'gs://{BUCKET_NAME}/{BUCKET_X1_TRAIN_DATA}'
    file = open(features_file, 'rb')
    features = pickle.load(file)
    file.close()

    captions_file = 'gs://{BUCKET_NAME}/{BUCKET_X2_TRAIN_DATA}'
    file = open(captions_file, 'rb')
    captions = pickle.load(file)
    file.close()

    print("data imported")
    return features, captions

def process_data(features, captions):
    sample = 1000

    cap_img_list = captions[0]

    X1 = []
    for cap_img in cap_img_list[:sample]:
        img_feature_matrix = features[cap_img][0]
        X1.append(img_feature_matrix)
    X1 = np.array(X1)

    X2 = np.array(captions[1]).astype(np.uint32)[:sample]

    vocab_size = 7589
    y = np.array([el[0] if len(el)>0 else vocab_size for el in captions[2][:sample]])

    print("data processed, ready for training")
    print("deleting full dataset in memory")
    features = None
    del features
    captions = None
    del captions
    return X1, X2, y

def train_model(X1, X2, y):
    model = LstmModel()
    print("model instantiated")
    trained_lstm = model.fit(X1, X2, y)
    print("model trained")
    return trained_lstm

def upload_model_to_gcp():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION)
    blob.upload_from_filename(JOBLIB_MODEL)

def save_model(model):
    joblib.dump(model, JOBLIB_MODEL)
    print("model saved locally")
    upload_model_to_gcp()
    print("model uploaded to gcp storage")

if __name__ == '__main__':
    features, captions = get_data()
    X1_train, X2_train, y_train = process_data(features, captions)
    trained_model = train_model(X1_train, X2_train, y_train)
    save_model(trained_model)
