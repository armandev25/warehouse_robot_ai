import tensorflow as tf
import numpy as np
import cv2

labels = ["fragile","hazardous","heavy"]

model = tf.keras.models.load_model("warehouse_classifier.h5")

img = cv2.imread("test.jpg")
img = cv2.resize(img,(224,224))
img = img/255.0
img = np.expand_dims(img,0)

pred = model.predict(img)
print("Prediction:", labels[np.argmax(pred)])
