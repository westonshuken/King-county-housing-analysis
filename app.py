#### IMPORTING LIBRARIES ####

import streamlit as st
import pickle
import numpy as np
import pandas as pd

#### LOADING FINAL RANDOM FOREST MODEL and SIMPLE VERSION OF RF MODEL (Less Features/more interpretable) ####

loaded_model = pickle.load(open('Models/final_rf.sav', 'rb'))
simple_model = pickle.load(open('Models/simple_rf.sav', 'rb'))

#### BRINGING IN THE TEST DATA ####

X_test = pd.read_csv('./Data/X_test_app.csv')
y_test =  pd.read_csv('./Data/y_test_app.csv')

#### HEADER ####
st.markdown('# King County Housing Price Predictor')

#### CUSTOM PREDICTION ####
st.header('Make a Custom Prediction')

basement = st.checkbox('Has a Basement')
if basement:
       has_base = 1
else:
       has_base = 0
view = st.checkbox('Has a Good View')
if view:
       has_view = 1
else:
       has_view = 0
waterfront = st.checkbox('Is a Waterfront Property')
if waterfront:
       on_water = 1
else:
       on_water = 0
sqft = st.slider('Total Living Sqft', 300, 10000)
bedrooms = st.slider('Bedrooms', 1, 8)
bathrooms = st.slider('Bathrooms', 1., 5., step=.5)
floors = st.slider('Floors', 1, 4)
distance = st.slider('Distance Major City (m)', 0, 300)
grade = st.slider('Grade Rating (quality)', 1, 10)

features =  [bedrooms, bathrooms, sqft, floors, on_water, has_view,
       grade, has_base, distance]
features_arr = np.array(features)

features_df = pd.DataFrame(data=[features_arr], columns=['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'view',
       'grade', 'basement', 'dist_major_cities'])

custom_predict = st.button('Predict Price')
if custom_predict:
       custom_prediction = simple_model.predict(features_arr.reshape(1, -1))
       col1, col2 = st.columns(2)
       with col2:
              st.write('###### Random Forest Regressor')  
              st.write('###### ~78% $R^2$')   
              st.write('###### ~140,000 RMSE') 
              st.write('##### Selcted Features:')
              st.dataframe(features_df.T)
       with col1:

              st.write('#### Price Prediction: $', str(int(custom_prediction)))

st.markdown('---')


#### HIGH PERFORMING MODEL PREDICTION ####
st.header('Model Prediction Example \n with standardized test data')
generate = st.button('Generate Prediction')

if generate:
       rint = np.random.randint(0, len(X_test))
       prediction = loaded_model.predict(X_test.iloc[rint].values.reshape(1, -1))
       col1, col2 = st.columns(2)

       with col2:
              st.write('###### Random Forest Regressor')  
              st.write('~87% $R^2$')   
              st.write('~107,000 RMSE') 
              st.write('Data:') 
              st.dataframe(X_test.iloc[rint].T)
       with col1:
              st.write('#### Predicted Price: $', str(int(prediction[0])))
              st.write('#### Actual Price: $', str(int(y_test.iloc[rint].values)))
              st.write('#### Error: $', str(abs(int(prediction[0]) - int(y_test.iloc[rint].values))))

#### SIDEBAR ####


st.sidebar.markdown("## About")
st.sidebar.markdown("This is a housing price predictor based on \
King County Housing Data from 2014-2015 which contains 21,000 data points of house sales.")

st.sidebar.markdown("This model is not intended for *actual* use. This data is from 2014-2015 and only from King County listings \
       , therefore it would not be applicable for use in **any** current situation (unfortunately time travel is not possible).") 

st.sidebar.markdown("The `Custom Prediction` allows the user to generate housing data/features. \
       The features we chose for this are based on the features with the *largest coefficients* in our \
       Linear Model and the *highest importance* in our Random Forest models.") 

st.sidebar.markdown("The `Model Prediction Example` generates data that has been cleaned, feature engineered, \
and standardized. \
As a consequence of the standardization, the DataFrame returns coefficients that are not quite interpretable, \
but rather it showcases the number of variables involved in the prediction process.") 

st.sidebar.markdown("For more information, please see our [Github](https://github.com/westonshuken/King-county-housing-analysis).")

st.sidebar.markdown('###### Application by Weston Shuken') 
st.sidebar.markdown('###### Data Processing & ML Modeling by Czarina Luna, Hatice Drogen, Ross McKim, & Weston Shuken')

