import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#mapping of words index BACK TO WORDs
word_index=imdb.get_word_index()
#word_index
reverse_word_index={value: key for key,value in word_index.items()}


#load the pre trained mode
model=load_model('C:\Python\RNN_project\simple_rnn_imdb.h5')

# Step 2: Helper Functions

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


#prediction functions
#predictons function

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)
    sentiment="postive" if prediction[0][0]>0.5 else 'negetive'
    return sentiment,prediction[0][0]

import streamlit as st
##streamlit app
#streamlit app
st.title('Movie Review analysis')
st.write('Enter a movie review to classify it as positive or negetive')

#user_input

user_input=st.text_area('Movie review')

if st.button('classify'):
    preprocessed_input=preprocess_text(user_input)

    #make prediction
    prediction=model.predict(preprocessed_input)
    sentiment='positive' if prediction[0][0]>0.5 else 'negetive'

    st.write(f'Sentiment:{sentiment}')
    st.write(f'prediction score:{prediction[0][0]}')
else:
        st.write('please write a moview review')
