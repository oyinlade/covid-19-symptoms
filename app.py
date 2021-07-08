import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB
st.write("""
# Covid 19 Prediction App
This app predicts the chances of you having covid 19!

""")

st.sidebar.header('Symptoms')

# Collects user input features into dataframe
def user_input_features():
    cough = st.sidebar.selectbox('cough',('yes','no'))
    fever= st.sidebar.selectbox('fever',('yes','no'))
    sore_throat = st.sidebar.selectbox('sore_throat',('yes','no'))
    shortness_of_breath = st.sidebar.selectbox('shortness_of_breath',('yes','no'))
    head_ache = st.sidebar.selectbox('head_ache',('yes','no'))
    age_60_and_above = st.sidebar.selectbox('age_60_and_above',('yes','no'))
    test_indication= st.sidebar.selectbox('test_indication',('Abroad','Contact with confirmed', 'Other'))

    data = {'cough': cough,
                'fever': fever,
                'sore_throat': sore_throat,
                'shortness_of_breath': shortness_of_breath,
                'head_ache': head_ache,
                'age_60_and_above': age_60_and_above,
                'test_indication': test_indication}

    #features = pd.DataFrame(data, index=[0])
    #return features
input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phases
data = pd.read_csv('data1.csv')
data.drop(['test_date', 'gender'], axis = 1, inplace = True)
covid_result = data.drop(columns=['corona_result'])
df = pd.concat([input_df,covid_result],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['age_60_and_above','test_indication']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df  # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')
#st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('dc_model.pkl', 'rb'))

# Apply model to make predictions
a = ['positive', 'negative','other']
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
covid = np.array(['negative','positive','other'])
st.write(covid[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
