from datetime import datetime

import joblib
import pandas as pd
import pytz
from flask import Flask
from flask import request
from flask_cors import CORS
from termcolor import colored

from greeneye.gcp import download_model

app = Flask(__name__)
CORS(app)

PATH_TO_MODEL = "green_eye/tf_cloud_train_tar_456ec0c5_58e7_4fe3_8ac4_19a67cdff005"

pipeline_def = {'pipeline': joblib.load(PATH_TO_MODEL),
                'from_gcp': False}

@app.route('/')
def index():
    return 'OK'


# @app.route('/predict_fare', methods=['GET', 'POST'])
# def predict_fare():
#     """
#     Expected input
#         {"pickup_datetime": 2012-12-03 13:10:00 UTC,
#         "pickup_latitude": 40.747,
#         "pickup_longitude": -73.989,
#         "dropoff_latitude": 40.802,
#         "dropoff_longitude":  -73.956,
#         "passenger_count": 2}
#     :return: {"predictions": [18.345]}
#     """
#     inputs = request.get_json()
#     if isinstance(inputs, dict):
#         inputs = [inputs]
#     inputs = [format_input(point) for point in inputs]
#     # Here wee need to convert inputs to dataframe to feed as input to our pipeline
#     # Indeed our pipeline expects a dataframe as input
#     X = pd.DataFrame(inputs)
#     # Here we specify the right column order
#     X = X[COLS]
#     pipeline = pipeline_def["pipeline"]
#     results = pipeline.predict(X)
#     results = [round(float(r), 3) for r in results]
#     return {"predictions": results}


# @app.route('/set_model', methods=['GET', 'POST'])
# def set_model():
#     inputs = request.get_json()
#     # model_dir = FOLDER_MODEL_PATH
#     # model_dir = inputs["model_directory"]
#     # pipeline_def["pipeline"] = download_model(model_directory=model_dir, rm=True)
#     pipeline_def["pipeline"] = download_model(rm=True)
#     pipeline_def["from_gcp"] = True
#     return {"reponse": f"correctly got model from {model_dir} directory on GCP"}


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)