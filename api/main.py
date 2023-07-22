from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=['*'],
    allow_methods=[]
)

model = load_model('emotions-cnn')
classes = ['happy', 'sad', 'angry', 'fearful']

@app.post('/predict')
async def predict(img: UploadFile = File(...)):
    contents = await img.read()
    image = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(np.array(image), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, 0)

    prediction = model.predict(image)
    score = tf.nn.softmax(prediction[0])
    prediction_class = classes[np.argmax(score)]

    return prediction_class