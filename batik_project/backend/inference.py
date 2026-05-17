import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os
from io import BytesIO
from PIL import Image


the_model = tf.keras.models.load_model('batik_model.keras')

CLASS_NAMES = ['Batik Kawung', 'Batik Megamendung', 'Batik Parang']
def predict_image(image):
    #Decode and resize to 224x224
    img = Image.open(BytesIO(image))
    resized_img =  img.resize((224, 224))
    
    #Apply preprocess_input
    img_array = np.array(resized_img)
    img_array = np.expand_dims(img_array, 0)
    cm_normalized = preprocess_input(img_array)

    #Run model.predict()
    raw_predictions = the_model.predict(cm_normalized)
    #Apply softmax (model outputs raw logits)
    probabilities = tf.nn.softmax(raw_predictions).numpy()

    predicted_index = np.argmax(probabilities)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[0][predicted_index])

    return {"class": predicted_class, "confidence": confidence}