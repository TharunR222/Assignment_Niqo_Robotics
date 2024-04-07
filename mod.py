import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model

st.title("Find fonts on the go!!")

data = st.file_uploader(label = "Upload the image containing the font")

if data:
  model = load_model('data_binaryfont_classification_model.h5')

  img = cv2.imread(data)
  resize = tf.image.resize(img, (400, 900))
  plt.imshow(resize.numpy().astype(int))

  yhat = model.predict(np.expand_dims(resize, 0))
  
  if yhat <= 0.7:
    st.write("Open Sans")
  else:
    st.write("Roboto")