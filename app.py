# from os import unsetenv
import streamlit as st
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')
import pickle

# loading the model and vecorizer
with open('finalized_model.pkl', 'rb') as f:
    cv, nv = pickle.load(f)


st.title('Sentimant Analysis on Product Reviews')

st.markdown('Welcome to our project in Sentimant Analysis, this project focuses on analyzing and predicting the reviews of the books from the "Al-Shaikh Book Store".')

st.markdown('Please enter a review so the model can predict a rating!')
# inputing a string to predict
z = st.text_input('Enter a Review')

# cleaning the review before throwing it in the model
z = re.sub('[^a-zA-Z]',' ',z)
z = z.lower()
z = z.split()

z1 = []

# more cleaning using nltk this time
for word in z:
    if word not in stopwords.words('english'):
        z1.append(word)
    if word in (['not','very','more','wasnt']):
        z1.append(word)

s = PorterStemmer()
z2=[]

# stemming the words by trnasforming them back to their roots
for i in z1:
    v=s.stem(i)
    z2.append(v)
    

z = ' '.join(z2)
z = np.array([z])
z1 = cv.transform(z)

rating = nv.predict(z1)

if z:
    st.markdown(f'Predicted Rating: {rating[0]}')
