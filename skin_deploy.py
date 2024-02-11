import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import keras

@st.cache(allow_output_mutation=True)
# Load your pre-trained model
def load_model():
	model = tf.keras.models.load_model('Epidermal_Disease.h5')
	return model
with st.spinner('Loading Model into Memory....'):
    model = load_model()
# model = keras.models.load_model('Epidermal_Disease.h5')

# Define the skin disease categories
disease_categories = ['Acne and Rosacea Photos', 'Eczema Photos', 'Exanthems and Drug Eruptions', \
                      'Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases', \
                      'Poison Ivy Photos and other Contact Dermatitis', 'Psoriasis pictures Lichen Planus and related diseases', \
                      'Scabies Lyme Disease and other Infestations and Bites', \
                      'Systemic Disease', 'Warts Molluscum and other Viral Infections']

# st.title("Skin Disease Classification")

# Upload an image
# image = st.file_uploader("Upload an image", type=["jpg", "png"])
# model = tf.keras.models.load_model('Epidermal_Disease.h5')
def predict_class(image, model):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [224,224])

	image = np.expand_dims(image, axis = 1 )

	prediction = model.predict(image)

	return prediction

# pred_probs = model.predict(uploaded_image)
# pred_probs[0]
# pred_classes = pred_probs.argmax(axis =1)
# print(pred_classes[0])
# print(disease_categories[pred_classes[0]])



st.image('logo_skin.jpg', width=100 )
st.title('Epidermal Disease Classifier')

file = st.file_uploader("Upload an image of a disease", type=["jpg", "png"])

if file is None:
    st.text('Waiting for upload...')
    
else:
    slot = st.empty()
    slot.text('Running the Inference...')
    test_image = Image.open(file)
    st.image(test_image, caption='Input Image', width= 500)
    pred = predict_class(test_image, model)
    # score = tf.nn.softmax(pred[0])
    # pred_probs = model.predict(test_image)
    # pred_probs[0]
    pred_classes = pred.argmax(axis=0)
    # print(pred_classes[0])
    # print(disease_categories[pred_classes[0]])
    output = "This image most likely belongs to {} .".format(disease_categories[pred_classes[0]])
    st.success(output)
