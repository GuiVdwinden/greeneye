import json
import os
import joblib
import tensorflow as tf
from tensorflow.keras import models
from google.cloud import storage
from google.oauth2 import service_account
from termcolor import colored
#from TaxiFareModel.params import BUCKET_NAME, PROJECT_ID, MODEL_NAME, MODEL_VERSION
from greeneye.params import PROJECT_NAME, BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, BUCKET_TRAINING_FOLDER, MODEL_PATH, MODEL_VERSION
from tensorflow.keras.models import load_model
import h5py
import gcsfs
import sys
from google.protobuf import text_format

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/Gui/Documents/gcp_keys/Batch487_Gui_LeWagon-538bf8f6a382.json"

def get_credentials():
    credentials_raw = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if '.json' in credentials_raw:
        credentials_raw = open(credentials_raw).read()
    creds_json = json.loads(credentials_raw)
    creds_gcp = service_account.Credentials.from_service_account_info(creds_json, 
        scopes=[u'https://www.googleapis.com/auth/cloud-platform'])

    return creds_gcp


# def storage_upload(model_version=MODEL_VERSION, bucket=BUCKET_NAME, rm=False):
#     client = storage.Client().bucket(bucket)

#     storage_location = 'models/{}/versions/{}/{}'.format(
#         MODEL_NAME,
#         model_version,
#         'model.joblib')
#     blob = client.blob(storage_location)
#     blob.upload_from_filename('model.joblib')
#     print(colored("=> model.joblib uploaded to bucket {} inside {}".format(BUCKET_NAME, storage_location),
#                   "green"))
#     if rm:
#         os.remove('model.joblib')

# def download_model():

#     # creds = get_credentials()
#     # client = storage.Client(credentials=creds, project=PROJECT_ID).bucket(bucket)
    
#     model = models.load_model('gs://green_eye/tf_cloud_train_tar_e2334b19_cf30_4382_98b6_fee4b8027abe')

#     return model



def download_model():
#    CREDENTIALS = get_credentials()
    MODEL_PATH = 'gs://green_eye/models/GreenEyeResNet50V1'

#    FS = gcsfs.GCSFileSystem(project=PROJECT_NAME,
#                             token=CREDENTIALS)
    
#    with FS.open(MODEL_PATH, 'rb') as model_file:
#         model_gcs = h5py.File(model_file, 'r')
#         myModel = load_model(model_gcs)

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
       myModel = tf.compat.v1.saved_model.loader.load(
           sess,
           [tf.compat.v1.saved_model.tag_constants.SERVING],MODEL_PATH)

    return myModel

print(download_model())

# def download_model(model_version=MODEL_VERSION, bucket=BUCKET_NAME, rm=True):
#     creds = get_credentials()
#     client = storage.Client(credentials=creds, project=PROJECT_ID).bucket(bucket)

#     storage_location = 'models/{}/versions/{}/{}'.format(
#         MODEL_NAME,
#         model_version,
#         'model.joblib')
#     blob = client.blob(storage_location)
#     blob.download_to_filename('model.joblib')
#     print(f"=> pipeline downloaded from storage")
#     model = joblib.load('model.joblib')
#     if rm:
#         os.remove('model.joblib')
#     return model
