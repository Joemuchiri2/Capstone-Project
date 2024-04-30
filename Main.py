import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title='Grocery Store Forecasting Challenge For Azubian',
    layout='wide',
    page_icon='📊'
)

# Project Title
st.title('Grocery Store Forecasting Challenge For Azubian')

#Read the Image
img = Image.open(".\image.png")
st.image(img, width=None, use_column_width=True)

# Read the CSV file and assign it to the 'data' variable
try:
    data = pd.read_csv('data.csv')
except FileNotFoundError:
    st.error("Dataset file not found.")
    data = None  # Assign None if file is not found to avoid NameError

# Button to view the dataset
if st.button('View Dataset'):
    if data is not None:
        st.write(data)


if data is not None:
             st.header('* The sales data is spanning from December 30, 1900, to August 10, 1902, the data is quite old,')



 # Sidebar inputs
st.sidebar.title('Filters')
Store = st.sidebar.selectbox('Select Store ID', data['store_id'].unique())
Holiday = st.sidebar.selectbox('Select Holiday', ['All', 'Holiday', 'RegularDay'])
    

feature = 'target'
feature2 = 'nbr_of_transactions'

# Filter the data
filtered_data = data[data['store_id'] == Store]
if Holiday == 'Holiday':
    filtered_data = filtered_data[filtered_data['is_holiday'] == 1]
elif Holiday == 'RegularDay':
    filtered_data = filtered_data[filtered_data['is_holiday'] == 0]

# Display features in separate columns
col1, col2 = st.columns(2)

with col1:

# Display feature 1
    st.subheader(f"Feature: Sales")

    # Display Total Sales
    total_sales = filtered_data['target'].sum()
    st.markdown(f'<span style="color: green;">Total Sales: {total_sales}</span>', unsafe_allow_html=True)
    fig = px.line(filtered_data, x='date', y=feature, title=f'Sales for: {Store}')
    st.plotly_chart(fig, use_container_width=True)

with col2:

# Display feature 2
    st.subheader(f"Feature: Transactions")

# Display Total Transaction 
    total_transactions = filtered_data['nbr_of_transactions'].sum()
    st.markdown(f'<span style="color: green;">Number of Transactions: {total_transactions}</span>', unsafe_allow_html=True)
    fig2 = px.line(filtered_data, x='date', y=feature2, title=f'Transaction for : {Store}')
    st.plotly_chart(fig2, use_container_width=True)


features, City, store_type = st.tabs(["FEATURES", "CITY", "STORE TYPE"]) 


with features:
    st.header('These are Features that are in the dataset and those that I had to engineree')

    st.markdown('*  date: The date of the transaction.')
    st.markdown('*  store_id: The ID of the store where the transaction occurred.')
    st.markdown('*  category_id: The category ID of the product.')
    st.markdown('*  target: The target variable, possibly the number of products purchased or sales amount.')
    st.markdown('*  onpromotion: Indicates whether the product was on promotion (1 for yes, 0 for no).')
    st.markdown('*  nbr_of_transactions: The number of transactions.')
    st.markdown('*  store_id: The ID of the store.')
    st.markdown('*  city: The city where the store is located.')
    st.markdown('*  type: The type of store.')
    st.markdown('*  cluster: The cluster to which the store belongs.')

with City:

 # Calculate total sales  for each City
    City_groups = data.groupby('city')['target'].sum().reset_index()

# Create a bar chart
    fig = px.bar(City_groups, x='city', y='target', labels={'city': 'City', 'target': 'Total Sale'},
             title='Total Sales by Cities', text='target')
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    st.plotly_chart(fig)


with store_type:

 # Calculate total sales  for each store type
    Store_sales = data.groupby('store_type')['target'].sum().reset_index()

# Create a bar chart
    fig = px.bar(Store_sales, x='store_type', y='target', labels={'store_type': 'Store Type', 'target': 'Total Sales'},
                  title='Total Sales by Store Type')
    fig.update_xaxes(categoryorder='total ascending')  # Sort stores by total sales 
    fig.update_traces(texttemplate='%{y}', textposition='outside')
    st.plotly_chart(fig)


Trends, Resampling = st.tabs(["TREND & SEASONALITY", "RESAMPLING"]) 

with Trends:
       
     # Display unique cities in a dropdown
     selected_city = st.selectbox('Select City', data['city'].unique())

# Filter the data based on the selected city
     filtered_data = data[data['city'] == selected_city]

     # Convert selected_city to string
     selected_city_str = str(selected_city)

# Plot filtered data
     fig = px.line(filtered_data, x='date', y='target', title='Target vs. Date by Category for ' + selected_city)
     st.plotly_chart(fig)

with Resampling: 
     st.markdown('hello') 



# Set the path to the folder containing your model files
model_path = "model_files"  # Change this to the correct folder path

# Define model file name
model_file = os.path.join(model_path, "best_model.joblib")

# Load the primary model or fallback to a default if not found
try:
    model = joblib.load(model_file)  # Load the specified model
except FileNotFoundError:
    st.warning(f"Model file '{model_file}' not found. Loading a default DecisionTreeRegressor as fallback.")
    model = DecisionTreeRegressor()  # Default model if the primary one is not found

# Load the encoder and scaler files
try:
    encoder = joblib.load(os.path.join(model_path, "encoder.joblib"))
    scaler = joblib.load(os.path.join(model_path, "scaler.joblib"))
except FileNotFoundError:
    st.error("Encoder or Scaler file not found. Ensure the correct paths and files exist.")

# Continue with the Streamlit app logic
st.title("Grocery Store Forecasting App")

city = st.text_input("City")
store_id = st.text_input("Store ID")
onpromotion = st.selectbox("On Promotion?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
is_holiday = st.selectbox("Holiday?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
category_id = st.text_input("Category ID")

# Generate the forecast if all inputs are provided
if st.button("Generate Forecast"):
    if not city or not store_id or not category_id:
        st.warning("Please provide all required inputs.")
    else:
        # Prepare the input data
        input_data = pd.DataFrame({
            "city": [city],
            "store_id": [store_id],
            "onpromotion": [onpromotion],
            "is_holiday": [is_holiday],
            "category_id": [category_id]
        })

        # Apply encoder and scaler (make sure they were loaded correctly)
        if 'encoder' in locals() and 'scaler' in locals():
            encoded_input = encoder.transform(input_data)
            scaled_input = scaler.transform(encoded_input)

            # Make predictions
            forecast = model.predict(scaled_input)

            # Display forecast results
            st.header("Forecast for the next eight weeks:")
            forecast_df = pd.DataFrame({
                "Week": np.arange(1, 9),
                "Predicted Sales": forecast.flatten()  # Flatten if prediction is 2D
            })
            st.table(forecast_df)

            # Visualize the forecast with a plot
            st.plotly_chart(px.line(forecast_df, x="Week", y="Predicted Sales", title="Forecasted Sales for the Next Eight Weeks"))




