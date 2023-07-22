import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2
import os

classes = ['happy', 'sad', 'angry', 'fearful']

def init():
    global model
    model_path = os.path.join(os.environ.get('AZUREML_MODEL_DIR'), 'emotions-cnn')

    model = load_model(model_path)

def run(img):
    contents = img.read()
    image = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(np.array(image), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, 0)

    prediction = model.predict(image)
    score = tf.nn.softmax(prediction[0])
    prediction_class = classes[np.argmax(score)]

    return prediction_class