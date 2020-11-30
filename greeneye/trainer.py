import datetime
import os
from google.cloud import storage
import pandas as pd
from sklearn import linear_model
import numpy as np
import tensorflow as tf
import tensorflow_cloud as tfc
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
import PIL.Image
from tensorflow.keras import models, layers, preprocessing, backend
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

PROJECT_NAME='Batch 487 - Le Wagon'

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME='green_eye'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
BUCKET_TRAIN_DATA_PATH = 'data/train_classes.csv'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

BUCKET_TRAINING_FOLDER = 'data/train-jpg'

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_PATH = 'models'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def get_data():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    client = storage.Client()
    train_classes = pd.read_csv("gs://{}/{}".format(BUCKET_NAME, BUCKET_TRAIN_DATA_PATH), nrows=1000)
    
    labels = []
    for i in train_classes['tags']:
        for x in i.split(' '):
            if x not in labels:
                labels.append(x)

    train_classes['tags'] = train_classes['tags'].apply(lambda x: x.split(' '))

    train_classes['image_name'] = train_classes['image_name'].apply(lambda x: (x + '.jpg') if '.jpg' not in x else x)

    return train_classes

def preprocess(df):    

    labels = ['haze','primary','agriculture','clear',
    'water','habitation','road','cultivation','slash_burn','cloudy',
    'partly_cloudy','conventional_mine','bare_ground','artisinal_mine',
    'blooming','selective_logging','blow_down']

    train = df.sample(frac=0.7,random_state=42)
    test = df.drop(train.index)

    datagen = preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.25)
    train_gen = datagen.flow_from_dataframe(dataframe=train, 
                    directory=("raw_data/train-jpg"), x_col="image_name", y_col="tags", subset="training", seed=42,
                    shuffle=True, class_mode="categorical", target_size=(224,224), batch_size=32, classes=labels)
    valid_gen = datagen.flow_from_dataframe(dataframe=train, 
                    directory=("raw_data/train-jpg"), x_col="image_name", y_col="tags", subset="validation", seed=42,
                    shuffle=True, class_mode="categorical", target_size=(224,224), batch_size=32, classes=labels)

    datagen_test = preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_gen = datagen_test.flow_from_dataframe(dataframe=test, 
                    directory=("raw_data/train-jpg"), x_col="image_name", y_col="tags",
                    class_mode="categorical", target_size=(224,224), batch_size=32, classes=labels)

    return train_gen, valid_gen, test_gen

# def fbeta(y_true, y_pred):
#     beta_squared = 4

#     tp = backend.sum(y_true * y_pred) + backend.epsilon()
#     fp = backend.sum(y_pred) - tp
#     fn = backend.sum(y_true) - tp

#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)

#     result = (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + backend.epsilon())

#     return result

def train_model(train_gen, valid_gen, test_gen):
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(224,224,3),
        pooling='avg')

    x = base_model.output

    # let's add a fully-connected layer
    x = layers.Dense(2048, activation='relu')(x)

    # and a logistic layer -- let's say we have 200 classes
    predictions = layers.Dense(17, activation='softmax')(x)

    # this is the model we will train
    model = models.Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
    STEP_SIZE_VALID=valid_gen.n//valid_gen.batch_size
    STEP_SIZE_TEST=test_gen.n//test_gen.batch_size
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_gen,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=1
    )

    return model

def save_model(model):

    save_path = os.path.join("gs://", BUCKET_NAME, "models")

    model.save(save_path)

    tfc.run(requirements_txt="requirements.txt",
        docker_image_bucket_name=BUCKET_NAME
        )

    MODEL_PATH = "models"
    checkpoint_path = os.path.join("gs://", BUCKET_NAME, MODEL_PATH, "save_at_{epoch}")
    tensorboard_path = os.path.join(
        "gs://", BUCKET_NAME, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path),
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
    ]

    client = storage.Client().bucket(BUCKET_NAME)

    storage_location = '{}/{}/{}/{}'.format(
        'models',
        MODEL_PATH,
        MODEL_VERSION,
        'ResNet50_test')
    blob = client.blob(storage_location)
    blob.upload_from_filename(local_model_name)
    print("uploaded model to gcp cloud storage under \n => {}".format(storage_location))

# df_ = get_data()
# print('got the data')
# train_gen_, valid_gen_, test_gen_ = preprocess(df_)
# print('preprocessed')
# print(train_model(train_gen_, valid_gen_, test_gen_))
# print('done training')

if __name__ == '__main__':
    # get training data from GCP bucket
    df = get_data()

    # preprocess data
    train_gen, valid_gen, test_gen = preprocess(df)

    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)
    model = train_model(train_gen, valid_gen, test_gen)

    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model(model)
