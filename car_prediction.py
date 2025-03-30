import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
cars_data = pd.read_csv('Cardetails.csv')

# Drop unnecessary column
cars_data.drop(columns=['torque'], inplace=True)

# Remove null values
cars_data.dropna(inplace=True)

# Remove duplicates
cars_data.drop_duplicates(inplace=True)

# Extract brand names from car names
def get_brand_name(car_name):
    return car_name.split(" ")[0].strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Encoding categorical variables
cars_data['name'].replace(['Maruti','Skoda','Honda','Hyundai','Toyota','Ford','Renault','Mahindra',
'Tata','Chevrolet','Datsun','Jeep','Mercedes-Benz','Mitsubishi','Audi',
'Volkswagen','BMW','Nissan','Lexus','Jaguar','Land','MG','Volvo','Daewoo',
'Kia','Fiat','Force','Ambassador','Ashok','Isuzu','Opel'],
                          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
                          ,inplace=True)

cars_data['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
cars_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
cars_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
cars_data['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5], inplace=True)

# Split data into input and output variables
input_data = cars_data.drop(columns=['selling_price'])
output_data = cars_data['selling_price']

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2)

# Train Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Test prediction
predict = model.predict(x_test)

# Predict on sample input
input_data_model = pd.DataFrame([[10,2015,10000,1,1,1,1,18.3,1991,147.9,5.0]],
                                columns=["name","year","km_driven","fuel","seller_type","transmission","owner","mileage","engine","max_power","seats"])
print(model.predict(input_data_model))


