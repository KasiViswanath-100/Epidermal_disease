import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model

@st.cache_resource
def load_model():
	model = tf.keras.models.load_model('Epidermal_Disease.h5')
	return model
with st.spinner('Loading Model into Memory....'):
    model = load_model()
disease_categories = ['Acne and Rosacea Photos', 'Eczema Photos', 'Exanthems and Drug Eruptions', \
                      'Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases', \
                      'Poison Ivy Photos and other Contact Dermatitis', 'Psoriasis pictures Lichen Planus and related diseases', \
                      'Scabies Lyme Disease and other Infestations and Bites', \
                      'Systemic Disease', 'Warts Molluscum and other Viral Infections']

def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img)
    # img = np.expand_dims(img, axis=0)
    return img

st.image('logo_skin.jpg', width=100 )
st.title('Epidermal Disease Classifier')
st.write('Upload an image for classification')

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)    
    processed_image = preprocess_image(image)    
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction,axis=0)
    print("Predicted class:", predicted_class)
    output = "This image most likely belongs to {} .".format(disease_categories[predicted_class[0]])
    # st.write('Prediction:')
    st.success(output)
