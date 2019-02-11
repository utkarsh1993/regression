#!/usr/bin/env python
# coding: utf-8

# # Importing Required Libraries

# In[19]:


from collections import Counter
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import os as os
import pickle
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
# from fancyimpute import KNN
from statistics import *
#from sklearn.linear_model import Ridge
import numpy as np
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import json
import sys


# In[20]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# In[21]:


from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[22]:


#path='C:\\Users\\utktripa1\\Desktop\\ajay\\req_server\\Regression\\'


# In[23]:


# Function Explore the data: Number of Rows, Columns, Data Types of Columns, Printing the top and bottom 5 and random sample of 10 rows
def explore(df):
    print('The input dataset df','has',df.shape[0],"rows and",df.shape[1],'columns') #Printing Shape
    print("The summary of Numeric Columns as desired:")
    print(df.describe())
    print("The top 5 rows are ")
    print(df.head()) #Prints top 5 rows
    print("The bottom 5 rows are ")
    print(df.tail()) #Prints bottom 5 rows
    print("The random 10 rows  are ")
    print(df.sample(10)) # Print randon 10 rows from data
    print("The type of variables in the input dataset are ")
    print(df.dtypes) # Prints Columns Data Type
    print("Null values in each column")
    print(df.isnull().sum()) # Checking Number of Null values in each columns
    print("Number of unique values in each Column")
    print(df.nunique()) # Checking Number of unique Values in each columns


# In[24]:

# Split Categorical and Continuous Variables in a Dataframe based on Number of Unique values present in Columns
def val_count(df, unique_count):
    temp = {"cat":[], "cont":[]} # Blank Dictionary
    for i in list(df.columns):
        if df[i].nunique() <= unique_count: #Comparing number of unique values in each column with desired value to decide if variable is Ctaegorical or Continuous
            temp["cat"].append(i)
        else:
            temp["cont"].append(i)
    
    df_cat = df[temp["cat"]] # Creating Separate Data frame of Categorical Variables
    df_cont = df[temp["cont"]] # Creating Data frame of Continuous Varables
    return df_cat, df_cont


# This function treates missing values of Continuous data with the help of above mentioned Imputation Techniques
def missing_val_treatment_cont(df, df_cont, way):
    if way == "drop":
        df.dropna(inplace = True)# iF DROP is selected, just drop any row or column with Null values and return whole data frame
        return df
    elif way == "mean": # Replace all Null values by column means (Valid only for Numeirc Variables)
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(df_cont)#Fit Imputer to data
        X = pd.DataFrame(imp.transform(df_cont), columns= df_cont.columns, index= df_cont.index)# Fill the NULL values
    elif way == "median":# Replace all Null values by column medians (Valid only for Numeirc Variables)
        imp = Imputer(missing_values='NaN', strategy='median', axis=0)
        imp.fit(df_cont)
        X = pd.DataFrame(imp.transform(df_cont), columns= df_cont.columns, index= df_cont.index)
    elif way == "ffill":# Replace all Null values by Next value present
        X = df_cont.fillna(method = "ffill")
    elif way == "bfill":# Replace all Null values by Previous value present
        X = df_cont.fillna(method = "bfill")  
    return X


# In[25]:


# This function uses fancyimpute library to Impute missing values of Numeric Data. The fancy impute uses machine learning algorithms to impute missing values.
def impute_fancy(df, way):#Only valid for numeric Data
    if way == "knn":# If you want to fill missing values using "KNN" algorithm
        df_numeric = df2.select_dtypes(include=[np.float])
        df_filled = pd.DataFrame(KNN(5).complete(df_numeric.as_matrix()), columns= df_numeric.columns, index= df_numeric.index)
    
    elif way == "mice": #If you want to fill missing values using "MICE" algorithm
        df_numeric = df2.select_dtypes(include=[np.float])
        df_filled = pd.DataFrame(mice.complete(df_numeric.as_matrix()), columns= df_numeric.columns, index= df_numeric.index)
    
    return df_filled


# In[26]:


# This function accepts Dataframe and Columns which needs to be scaled and return the full data frame with scaled variables
#Pass Columns in form of list
# data -: Dataframe, 
# Cols :list of column names whichneeds scaling
def scale_data(data, cols, way):     
    if way == "min_max":# If min max scaling is used in "way" argumtn
        scale = MinMaxScaler()
        temp = scale.fit_transform(data[cols].values)
        temp = pd.DataFrame(temp, columns= cols, index= data.index)
        data[cols] = temp[cols]
    elif way == "standard":# If Standard Scaling is passed in "way" argument
        scale = StandardScaler()
        temp = scale.fit_transform(data[cols].values)
        print(scale.mean_)
        temp = pd.DataFrame(temp, columns= cols, index= data.index)
        data[cols] = temp[cols]
    return data


# In[27]:


# This function is used to construct polynomial for all possible combinations upto degree as desired
# data -: Data frame
def poly_features(data):
    poly = PolynomialFeatures(degree=2)# Creating the Object
    X = poly.fit_transform(data)# Transform the data and generate new features
    return X


# In[28]:


# This function takes categorical Columns, Creates Dummy Variable
# Returns a single and final data frame
def dummy_creation(df):
    new_df = pd.get_dummies(df, drop_first= True)
    return new_df


# In[29]:


def model_load(name):
    model_pkl = open(name, 'rb')
    model = pickle.load(model_pkl)
    print("Loaded model :: ", model)
    return model

model = model_load("model.pkl")
# In[30]:


# # Deploying Using Flask
 
# Flask, a microframework for building websites and APIs in Python, to build our Web API and how to persist our model so we can have access to it without always having to retrain it each time we want to make a prediction

app = Flask(__name__)
@app.route('/predict_reg', methods=['GET','POST'])

def predict_reg():
    
    '''Reading the Input of Request File. Request will always be in form of json since POST method is used.
       Here onwards prediction Starts for Test Set or Data Provided by User
    '''
    
    ## Reading JSON file from request
    data = request.get_json(force = True)
    ## loading the data in json format
    jdata = json.loads(data)
    ## Creating the Data frame from Json
    data = pd.DataFrame(jdata)
    print(data.columns)
    print(data.head())
    ## Processing the new Data as Required
    #Creating a dictionary which contains names of Categorical and Continuous Variables
    df_cat, df_cont = val_count(data,10)
    # Filling missing values in continuous data
    df_cont = missing_val_treatment_cont(data, df_cont, "median")
    # Scaling the Desired Numeric Columns in Data
    # df_cont = scale_data(df_cont, list(df_cont.columns), "standard")
    #Creating Dummy Variables
    df_cat = dummy_creation(df_cat)
    # Concating both data frames to get final Data
    final_data = pd.concat([df_cont, df_cat], axis=1)
    ## Predicting the value Of Response Variable and Probabilties
    data = pd.DataFrame({"prediction" : list(model.predict(final_data))}) 
    
    '''
    Server will send the below 2 additonal columns in form of dictionary consisting of Probability of being 1 and corresponding Prediction
    '''
    print(data.head())
    return data.to_json()


# In[ ]:


if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug=False)


# In[ ]:




