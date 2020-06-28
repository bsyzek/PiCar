import pickle
import boto3
import io
import numpy as np
from datetime import datetime

class AWSManager:

    TRAINING_DATA_BUCKET = "pi-car-training-data"

    def __init__(self):
        pass

    def upload_training_data(self, images, labels):
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(self.TRAINING_DATA_BUCKET)

        training_images = np.array(images)
        training_labels = np.array(labels)
        training_data = {"images": training_images, "labels": training_labels}
       
        data_stream = io.BytesIO()
        pickle.dump(training_data, data_stream)
        data_stream.seek(0)

        file_name = self.create_file_name()
        bucket.upload_fileobj(data_stream, file_name)

    def create_file_name(self):
        date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        file_name = date_time + "_train.pkl"
        return file_name
