import pickle as pk
import streamlit as st
import numpy as np
import pandas as pd
import transmission

model = pk.load(open('model-2.pkl' , 'rb'))
st.header("KRITI CAR BAZAAR")

cars_data = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
  car_name = car_name.split(" ")[0] # we wrote index 0 as every car name has brand at first
  return car_name.strip() # used strip to remove extra spaces\

cars_data['name'] = cars_data['name'].apply(get_brand_name)

st.selectbox("select car brand" , cars_data['name'].unique())
st.slider('Car manufactured Year',1994,2024)
st.slider('Number of Kilometers Driven',11,200000)
st.selectbox("Fuel Type" , cars_data['fuel'].unique())
st.selectbox("Seller Type" , cars_data['seller_type'].unique())
st.selectbox("Transmission Type" , cars_data['transmission'].unique())
st.slider('Car Mileage',10,40)
st.slider('Engine CC',700,5000)
st.slider('Max Power',0,200)
st.slider('Number of Seats',5,10)

if st.button("Predict"):
  input_data_model = pd.DataFrame([[name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seat]]
                                ,columns=["name","year","km_driven","fuel","seller_type","transmission","owner","mileage","engine","max_power","seats"])
  st.write(input_data_model)