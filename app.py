import streamlit as st
import pickle
import numpy as np
import pandas as pd

loaded_model = pickle.load(open('Model/final_rf.sav', 'rb'))
simple_model = pickle.load(open('Model/simple_rf.sav', 'rb'))

X_test = pd.read_csv('./Data/X_test_app.csv')
y_test =  pd.read_csv('./Data/y_test_app.csv')

st.header('King County Housing Price Predictor')
st.write('###### *View Sidebar to Make Predictions*') 
st.write('<---------------------------------') 

header = st.sidebar.image('./Images/daria-nepriakhina-LZkbXfzJK4M-unsplash.jpg')
st.markdown('---') 

st.sidebar.write('# Model Prediction Example \n with standardized test data')
generate = st.sidebar.button('Generate Prediction')

if generate:
       st.sidebar.write('MOBILE ONLY: Close sidebar to view prediction')
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
    


st.sidebar.markdown('---') 


st.sidebar.write('# Custom Prediction')

sqft = st.sidebar.slider('Total Living Sqft', 300, 10000)
bedrooms = st.sidebar.slider('Bedrooms', 1, 8)
bathrooms = st.sidebar.slider('Bathrooms', 1., 5., step=.5)
floors = st.sidebar.slider('Floors', 1, 4)
distance = st.sidebar.slider('Distance Major City (m)', 0, 300)
basement = st.sidebar.select_slider('Has Basement (0=no | 1=yes)', [0, 1])
view = st.sidebar.select_slider('Has Good View (0=no | 1=yes)', [0, 1])
waterfront = st.sidebar.select_slider('Waterfront Property (0=no | 1=yes)', [0, 1])
grade = st.sidebar.slider('Grade Rating (quality)', 1, 10)

features =  [bedrooms, bathrooms, sqft, floors, waterfront, view,
       grade, basement, distance]
features_arr = np.array(features)

features_df = pd.DataFrame(data=[features_arr], columns=['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'view',
       'grade', 'basement', 'dist_major_cities'])

custom_predict = st.sidebar.button('Predict Price')
if custom_predict:
       st.sidebar.write('MOBILE ONLY: Close sidebar to view prediction')     
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

st.sidebar.markdown('---')

with st.sidebar.expander("Learn More"):
       st.markdown("This is a housing price predictor based off of \
       King County Housing Data from 2014-2015 which contains 21,000 data points of house sales")

       st.markdown("The `Model Prediction Example` generates data that our team cleaned the data, performed feature engineering \
       , and data standardized. \
       As a consequence of the standardization, the DataFrame returned is not interpretable in terms of the values, \
       but rather it showcases the number of variables involved in the prediction process.") 

       st.markdown("The `Custom Prediction` allows the user to generate home data in a simplistic manner. \
              The features we chose for this are based on the information with the largest coefficients in our \
              Linear Models and highest importance in our Random Forest models.") 

       st.markdown("This model is only intended for actual use. This data is from 2014-2015 in a specific county \
              , therefore it would only be applicable to use in this situation (unfortunately time travel is not possible).") 

       st.markdown("For more information, please see our [Github](https://github.com/westonshuken/King-county-housing-analysis).") 

