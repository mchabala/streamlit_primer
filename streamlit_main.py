import streamlit as st
import streamlit_antd_components as sac
import json
import folium
from folium.plugins import MarkerCluster
import itertools
import pandas as pd
import pdb
from datetime import date
import time
import os
import sys
import copy
import io
import altair as alt
from branca.element import Template, MacroElement
from datetime import datetime

from iris_random_forest import iris_random_forest

st.set_page_config(layout="wide", page_title="Streamlit Primer", page_icon=":coffee:")

def home_page():
    st.title('Home')
    
def data_upload_page():
    st.title('Data Upload/ Review')
    uploaded_file = st.file_uploader("Choose an Excel file", type = 'xlsx')
    
    # Try to load required sheets into session state. Throw error if not present
    required_columns = ['Petal_width', 'Petal_length', 'Sepal_width', 'Sepal_length', 'Species_name']
    df_columns_pretty = [name.replace("_", " ").title() for name in required_columns][:-1]
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        if all(df.columns.isin(required_columns)):
            st.session_state['data'] = df
            st.success("File successfully uploaded")
        else:
            st.error("Uploaded file is not in correct format")
    
    st.divider()
    st.header('Data Exploration')
    if 'data' not in st.session_state:
        st.warning('Upload a file to see an overview of the scenario')
        return False

    xcol, ycol, _ = st.columns(3)
    x_axis_col = xcol.selectbox("What would you like the x-axis to be?", df_columns_pretty, index = 3)
    y_axis_col = ycol.selectbox("What would you like the y-axis to be?", [i for i in df_columns_pretty if i != x_axis_col], index = 2)

    st.subheader(f"{x_axis_col} vs {y_axis_col}")
    st.scatter_chart(data=st.session_state['data'], x=required_columns[df_columns_pretty.index(x_axis_col)],
                     y=required_columns[df_columns_pretty.index(y_axis_col)], color='Species_name', height = 500)


def train_model_page():
    st.title('Train Model')
    
    if 'data' not in st.session_state:
        st.warning('Please upload a data file before training model')
        return False
    
    st.subheader("Model Parameters")
    
    test_size = st.slider("Testing set size", 0.01, .99)
    n_estimators = st.number_input("# Estimators", min_value= 1, max_value= 300, help = "The number of trees in the forest")
    criterion_options = ['gini', 'entropy', 'log_loss']
    criterion = st.selectbox("Criterion", criterion_options, help="The function to measure the quality of a split")
    
    
    if st.button("Run Model"):
        model, accuracy, feature_imp = iris_random_forest(st.session_state['data'], n_estimators, criterion, test_size)
        
        # Save to session state
        st.session_state['model'] = model
        st.session_state['accuracy'] = accuracy
        st.session_state['feature_imp'] = feature_imp
    
    if 'model' in st.session_state:
        st.subheader('Model Performance')
        st.metric('Model Accuracy', '{:.1%}'.format(st.session_state['accuracy']))
        st.dataframe(st.session_state['feature_imp'])
            
def predictions_page():
    st.title('Predictions')
    
    if 'model' not in st.session_state:
        st.warning('Please train a model first')
        return False
    
    petal_width = st.slider('Petal Width (cm)', 0.0, 10.0)
    petal_length = st.slider('Petal Length (cm)', 0.0, 10.0)
    sepal_width = st.slider('Sepal Width (cm)' , 0.0, 10.0)
    sepal_length = st.slider('Sepal Length (cm)', 0.0, 10.0)
    
    prediction = st.session_state['model'].predict([[petal_width, petal_length, sepal_width, sepal_length]])[0]
    st.header(f'Your predicted flower is {prediction}!')
    

# Page Navigation
with st.sidebar:

    col1,col2, col3 = st.columns([1,6,1]) # adjust the ratio as needed
    with col2:
        st.image("norm.png")

    # adding a visual separator
    st.sidebar.write("---")

    page = sac.menu(items = [
        sac.MenuItem('Home', icon='house-fill'),
        sac.MenuItem('Data Upload/Review', icon = "upload"),
        sac.MenuItem('Train Model', icon = "gear-fill"),
        sac.MenuItem('Predictions', icon = "bar-chart-fill")])

if page == 'Home':
    home_page()
if page == 'Data Upload/Review':
    data_upload_page()
if page == 'Train Model':
    train_model_page()
if page == 'Predictions':
    predictions_page()
