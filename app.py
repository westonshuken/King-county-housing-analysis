import streamlit as st
import pickle
import numpy as np
import pandas as pd

loaded_model = pickle.load(open('Model/final_rf.sav', 'rb'))
simple_model = pickle.load(open('Model/simple_rf.sav', 'rb'))

X_test = pd.read_csv('./Data/X_test_app.csv')
y_test =  pd.read_csv('./Data/y_test_app.csv')

st.header('King County Housing Price Predictor')
st.image('./Images/daria-nepriakhina-LZkbXfzJK4M-unsplash.jpg')
st.markdown('---') 

st.sidebar.write('# Model Prediction Example \n with standardized test data')
generate = st.sidebar.button('Generate Prediction')

if generate:
    st.write('###### Random Forest Regressor')  
    st.write('###### ~87% R2')   
    st.write('###### ~107,000 RMSE') 
    rint = np.random.randint(0, len(X_test))
    st.dataframe(X_test.iloc[rint].T)
    prediction = loaded_model.predict(X_test.iloc[rint].values.reshape(1, -1))
    st.write('#### Predicted Price: $', str(int(prediction[0])))
    st.write('#### Actual Price: $', str(int(y_test.iloc[rint].values)))
    st.write('#### Error: $', str(abs(int(prediction[0]) - int(y_test.iloc[rint].values))))

st.sidebar.markdown('---') 


st.sidebar.write('## Custom Prediction')

sqft = st.sidebar.slider('Total Living Sqft', 300, 10000)
bedrooms = st.sidebar.slider('Bedrooms', 1, 8)
bathrooms = st.sidebar.slider('Bathrooms', 1., 5., step=.5)
floors = st.sidebar.slider('Floors', 1, 4)
distance = st.sidebar.slider('Distance Major City (m)', 0, 300)
basement = st.sidebar.select_slider('Has Basement', [0, 1])
view = st.sidebar.select_slider('Has Good View', [0, 1])
waterfront = st.sidebar.select_slider('Waterfront Property', [0, 1])
grade = st.sidebar.slider('Grade Rating', 1, 10)

features =  [bedrooms, bathrooms, sqft, floors, waterfront, view,
       grade, basement, distance]
features_arr = np.array(features)

features_df = pd.DataFrame(data=[features_arr], columns=['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'view',
       'grade', 'basement', 'dist_major_cities'])

custom_predict = st.sidebar.button('Predict Price')
if custom_predict:
    st.write('###### Random Forest Regressor')  
    st.write('###### ~78% R2')   
    st.write('###### ~140,000 RMSE') 
    custom_prediction = simple_model.predict(features_arr.reshape(1, -1))
    st.write('#### Price: $', str(int(custom_prediction)))
    st.write('##### Selcted Features:')
    st.dataframe(features_df)




