import datetime
import os
from google.cloud import storage
import pandas as pd
from sklearn import linear_model
import numpy as np
import tensorflow as tf
import tensorflow_cloud as tfc
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
import PIL.Image
from tensorflow.keras import models, layers, preprocessing, backend
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from greeneye.params import PROJECT_NAME, BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, BUCKET_TRAINING_FOLDER, MODEL_PATH, MODEL_VERSION


def get_data():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    client = storage.Client()
    df = pd.read_csv("gs://{}/{}".format(BUCKET_NAME, BUCKET_TRAIN_DATA_PATH), nrows=1000)
    """download image folder from google cloud bucket"""
    return df

def preprocess(df):

    labels = ['haze','primary','agriculture','clear',
    'water','habitation','road','cultivation','slash_burn','cloudy',
    'partly_cloudy','conventional_mine','bare_ground','artisinal_mine',
    'blooming','selective_logging','blow_down']

    """create column tag_list with list of tags"""
    df['tag_list'] = df['tags'].apply(lambda x: x.split(' '))

    """add file extension to image_name column"""
    df['image_name'] = df['image_name'].apply(lambda x: (x + '.jpg') if '.jpg' not in x else x)

    """drop the one picture without a weather label"""
    index = df[df['clear'] + df['cloudy'] + df['haze'] + df['partly_cloudy'] != 1].index
    df.drop(index=index[0])

    """remove clear and haze labels from tag_list for better model accuracy"""
    def remove_label_from_tag_list(label):
        for i in df['tag_list']:
            try:
                i.remove(label)
            except ValueError:
                pass

    remove_label_from_tag_list('clear')
    remove_label_from_tag_list('haze')

    train = df.sample(frac=0.7,random_state=42)
    test = df.drop(train.index)

    img_size=128

    datagen = preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=0.2,
            horizontal_flip=True, vertical_flip=True, validation_split=0.25)
    train_gen = datagen.flow_from_dataframe(dataframe=train,
                    directory=("raw_data/train-jpg"), x_col="image_name", y_col="tag_list", subset="training", seed=42,
                    shuffle=True, class_mode="categorical", target_size=(img_size,img_size), batch_size=32)
    valid_gen = datagen.flow_from_dataframe(dataframe=train,
                    directory=("raw_data/train-jpg"), x_col="image_name", y_col="tag_list", subset="validation", seed=42,
                    shuffle=True, class_mode="categorical", target_size=(img_size,img_size), batch_size=32)

    datagen_test = preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_gen = datagen_test.flow_from_dataframe(dataframe=test,
                    directory=("raw_data/train-jpg"), x_col="image_name", y_col="tag_list",
                    class_mode="categorical", target_size=(img_size,img_size), batch_size=32, classes=labels)

    return train_gen, valid_gen, test_gen

"""create f2 score"""
def fbeta(y_true, y_pred, beta=2):
     bb = beta ** 2

     tp = backend.sum(y_true * y_pred) + backend.epsilon()
     fp = backend.sum(y_pred) - tp
     fn = backend.sum(y_true) - tp

     precision = tp / (tp + fp)
     recall = tp / (tp + fn)

     score = (bb + 1) * (precision * recall) / (bb * precision + recall + backend.epsilon())

     return score

def train_model(train_gen, valid_gen, test_gen):
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3),
        pooling='avg')

    x = base_model.output

    # let's add a fully-connected layer
    x = layers.Dense(2048, activation='relu')(x)

    # and a logistic layer -- let's say we have 200 classes
    predictions = layers.Dense(15, activation='sigmoid')(x)

    # this is the model we will train
    model = models.Model(inputs=base_model.input, outputs=predictions)

    # freeze base layers
    for layer in base_model.layers:
        layer.trainable = False

    optimizer = Adam(0.001, decay=0.0003)

    callbacks = [ModelCheckpoint('best_weights.hdf5', monitor='val_loss', save_best_only=True, verbose=2, save_weights_only=False),
             ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.0000001),
             EarlyStopping(monitor='val_loss', patience=5, verbose=0)]

    model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy', fbeta])

    STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
    STEP_SIZE_VALID=valid_gen.n//valid_gen.batch_size
    STEP_SIZE_TEST=test_gen.n//test_gen.batch_size
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_gen,
                        validation_steps=STEP_SIZE_VALID,
                        callbacks=callbacks
                        epochs=5)
    print(model)

    # unfreeze base layers and train entire model with new images
    optimizer = Adam(0.0001, decay=0.00000001)
    model.load_weights('best_weights.hdf5', by_name=True)
    for layer in base_model.layers:
        layer.trainable = True

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', fbeta])
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_gen,
                        validation_steps=STEP_SIZE_VALID,
                        callbacks=callbacks,
                        epochs=50)

    return model

def save_model(model):
#     tfc.run(
#         docker_image_bucket_name=BUCKET_NAME
#         )

#     save_path = os.path.join("gs://", BUCKET_NAME, MODEL_PATH)

# #    save_path = "gs://{}/{}".format(BUCKET_NAME, MODEL_PATH)

#     checkpoint_path = "gs://{}/{}".format(BUCKET_NAME, MODEL_PATH)
#     tensorboard_path = "gs://{}/{}".format(BUCKET_NAME, MODEL_PATH)

#     # checkpoint_path = os.path.join("gs://", BUCKET_NAME, MODEL_PATH, "save_at_{epoch}")
#     # tensorboard_path = os.path.join("gs://", BUCKET_NAME, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#     callbacks = [
#         tf.keras.callbacks.ModelCheckpoint(checkpoint_path),
#         tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1),
#         tf.keras.callbacks.EarlyStopping(monitor='categorical_crossentropy', patience=3),
#     ]
    model.save('gs://green_eye/models')
#    models.save_model(model, 'my_model.h5', save_format='h5')
#    !gsutil -m cp model.h5 gs://green_eye/models/model.h5

    print(model)


# def save_model(model):

#     save_path = os.path.join("gs://", BUCKET_NAME, "models")

#     tfc.run(requirements_txt="requirements.txt",
#         docker_image_bucket_name=BUCKET_NAME
#         )

#     MODEL_PATH = "models"
#     checkpoint_path = os.path.join("gs://", BUCKET_NAME, MODEL_PATH, "save_at_{epoch}")
#     tensorboard_path = os.path.join(
#         "gs://", BUCKET_NAME, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     )
#     callbacks = [
#         tf.keras.callbacks.ModelCheckpoint(checkpoint_path),
#         tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1),
#         tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
#     ]

#     model.save(save_path)

#     client = storage.Client().bucket(BUCKET_NAME)

#     storage_location = '{}/{}/{}/{}'.format(
#         'models',
#         MODEL_PATH,
#         MODEL_VERSION,
#         'ResNet50_test')
#     print("uploaded model to gcp cloud storage under \n => {}".format(storage_location))
    # blob = client.blob(storage_location)
    # blob.upload_from_filename(local_model_name)


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
