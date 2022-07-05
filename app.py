import tensorflow as tf

model = tf.keras.models.load_model('model/mkp02.h5')

import streamlit as st

font="century gothic"

st.write("""
        Monkey-pox detector by mHealth Lab, BME, BUET
        """)
st.write("Check on the go")
file = st.file_uploader("Please upload an image file", type=["jpg", "png","jpeg","bmp"])

import cv2
from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
    
        size = (256, 256)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(256, 256),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if prediction> 0.5:
        st.write("It might be Monkeypox. You should visit a specialist immediately. Thank you.")
        st.write((1-prediction))
    else:
        st.write("It's most probably not monkeypox, but still you should visit a skin specialist. Thank you.")
        st.write((1-prediction))
    
    
