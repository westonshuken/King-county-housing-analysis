import streamlit as st
import pickle
import numpy as np
import pandas as pd

loaded_model = pickle.load(open('Model/final_rf.sav', 'rb'))
simple_model = pickle.load(open('Model/simple_rf.sav', 'rb'))

X_test = pd.read_csv('./Data/X_test_app.csv')
y_test =  pd.read_csv('./Data/y_test_app.csv')

st.header('King County Housing Price Predictor')
st.write('###### *View Sidebar to Make Predictions (top-left corner)*') 
st.write('<---------------------------------') 

st.markdown('---') 

st.write('# Model Prediction Example \n with standardized test data')
generate = st.button('Generate Prediction')

if generate:
       st.write('MOBILE ONLY: Close sidebar to view prediction')
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
    


st.markdown('---') 


st.write('# Custom Prediction')

sqft = st.slider('Total Living Sqft', 300, 10000)
bedrooms = st.slider('Bedrooms', 1, 8)
bathrooms = st.slider('Bathrooms', 1., 5., step=.5)
floors = st.slider('Floors', 1, 4)
distance = st.slider('Distance Major City (m)', 0, 300)
basement = st.select_slider('Has Basement (0=no | 1=yes)', [0, 1])
view = st.select_slider('Has Good View (0=no | 1=yes)', [0, 1])
waterfront = st.select_slider('Waterfront Property (0=no | 1=yes)', [0, 1])
grade = st.slider('Grade Rating (quality)', 1, 10)

features =  [bedrooms, bathrooms, sqft, floors, waterfront, view,
       grade, basement, distance]
features_arr = np.array(features)

features_df = pd.DataFrame(data=[features_arr], columns=['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'view',
       'grade', 'basement', 'dist_major_cities'])

custom_predict = st.button('Predict Price')
if custom_predict:
       st.write('MOBILE ONLY: Close sidebar to view prediction')     
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

with st.expander("Learn More"):
       st.markdown("This is a housing price predictor based on \
       King County Housing Data from 2014-2015 which contains 21,000 data points of house sales.")

       st.markdown("The `Model Prediction Example` generates data that our team cleaned, feature engineered, \
       and standardized. \
       As a consequence of the standardization, the DataFrame returned is not interpretable in terms of the values, \
       but rather it showcases the number of variables involved in the prediction process.") 

       st.markdown("The `Custom Prediction` allows the user to generate home data in a simplistic manner. \
              The features we chose for this are based on the features with the largest coefficients in our \
              Linear Model and the highest importance in our Random Forest models.") 

       st.markdown("This model not  intended for actual use. This data is from 2014-2015 and only from King County listings \
              , therefore it would not be applicable to use in any current situation (unfortunately time travel is not possible).") 

       st.markdown("For more information, please see our [Github](https://github.com/westonshuken/King-county-housing-analysis).") 

