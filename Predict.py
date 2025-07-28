import tensorflow as tf
import numpy as np

def predict(model,img):
    img_array=tf.keras.preprocessing.image.img_to_array(img)
    img_array=tf.expand_dims(img_array,0)

    prediction=model.predict(img_array)
    
    prediction=np.argmax(prediction)

    return prediction

