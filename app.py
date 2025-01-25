import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


model = tf.keras.models.load_model('model.h5')
with open ('label_encoder_pre8.pkl','rb') as file:
    label_encoder_pre8=pickle.load(file)
with open ('label_encoder_pre9.pkl','rb') as file:
    label_encoder_pre9=pickle.load(file)
with open ('label_encoder_pre10.pkl','rb') as file:
    label_encoder_pre10=pickle.load(file)
with open ('label_encoder_pre11.pkl','rb') as file:
    label_encoder_pre11=pickle.load(file)
with open ('label_encoder_pre17.pkl','rb') as file:
    label_encoder_pre17=pickle.load(file)
with open ('label_encoder_pre19.pkl','rb') as file:
    label_encoder_pre19=pickle.load(file)
with open ('label_encoder_pre25.pkl','rb') as file:
    label_encoder_pre25=pickle.load(file)
with open ('label_encoder_pre30.pkl','rb') as file:
    label_encoder_pre30=pickle.load(file)
with open ('label_encoder_pre32.pkl','rb') as file:
    label_encoder_pre32=pickle.load(file)
with open ('label_encoder_risk1yr.pkl','rb') as file:
    label_encoder_risk1Yr=pickle.load(file)
with open ('one_hot_encoder_dgn.pkl','rb') as file:
    one_hot_encoder_dgn=pickle.load(file)
with open ('one_hot_encoder_pre6.pkl','rb') as file:
    one_hot_encoder_pre6=pickle.load(file)
with open ('one_hot_encoder_pre14.pkl','rb') as file:
    one_hot_encoder_pre14=pickle.load(file)
with open ('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

st.title('Thoracic Surgery Data - UCIML')

dgn = st.selectbox('DGN',one_hot_encoder_dgn.categories_[0])
pre6 = st.selectbox('PRE6',one_hot_encoder_pre6.categories_[0])
pre14 = st.selectbox('PRE14',one_hot_encoder_pre14.categories_[0])
age = st.slider('Age', 18, 99)
pre4 = st.number_input('PRE4')
pre5 = st.number_input('PRE5')
pre7 = st.selectbox('PRE7',[0,1])
pre8 = st.selectbox('PRE8',[0,1])
pre9 = st.selectbox('PRE9',[0,1])
pre10 = st.selectbox('PRE10',[0,1])
pre11= st.selectbox('PRE11',[0,1])
pre17 = st.selectbox('PRE17',[0,1])
pre19 = st.selectbox('PRE19',[0,1])
pre25= st.selectbox('PRE25',[0,1])
pre30 = st.selectbox('PRE30',[0,1])
pre32 = st.selectbox('PRE32',[0,1])


input_data = pd.DataFrame({
'PRE4':[pre4],
'PRE5':[pre5],
'PRE7':[pre7],
'PRE8':[pre8],
'PRE9':[pre9],
'PRE10':[pre10],
'PRE11':[pre11],
'PRE17':[pre17],
'PRE19':[pre19],
'PRE25':[pre25],
'PRE30':[pre30],
'PRE32':[pre32],
'AGE':[age]
})



dgn_encoded = one_hot_encoder_dgn.transform([[input_data['DGN']]]).toarray()
dgn_encoded_df = pd.DataFrame(dgn_encoded,columns=one_hot_encoder_dgn.get_feature_names_out(['DGN']))
pre6_encoded = one_hot_encoder_pre6.transform([[input_data['PRE6']]]).toarray()
pre6_encoded_df = pd.DataFrame(pre6_encoded,columns=one_hot_encoder_pre6.get_feature_names_out(['PRE6']))
pre14_encoded = one_hot_encoder_pre14.transform([[input_data['PRE14']]]).toarray()
pre14_encoded_df = pd.DataFrame(pre14_encoded,columns=one_hot_encoder_pre14.get_feature_names_out(['PRE14']))

input_data = pd.concat([input_data.drop("DGN",axis=1),dgn_encoded_df],axis=1)
input_data = pd.concat([input_data.drop("PRE6",axis=1),pre6_encoded_df],axis=1)
input_data = pd.concat([input_data.drop("PRE14",axis=1),pre14_encoded_df],axis=1)

input_df_scaled = scaler.transform(input_data)
input_df_scaled

prediction = model.predict(input_df_scaled)
prediction_proba = prediction[0][0]

st.write(f'Risk Probablity : {prediction_proba : .2f}')

if prediction_proba > 0.5 :
    st.write('Patient has low post operative survival rate in the first yr post surgery ')
else :
    st.write('Patient has high post operative survival rate in the first yr post surgery ')
