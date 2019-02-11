#!/usr/bin/env python
# coding: utf-8

# ## Orthogonal Data Science Project Process (ODSP)- Regression

# In[1]:


#from IPython.display import HTML
#from IPython.display import Image


# Import HTML and IMage from IPython

# # Importing Required Libraries

# In[31]:


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


# In[3]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# In[4]:


from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[6]:


# Function to change Working Directory and list elements present in new working directory
def set_wd(wd):
    print(os.getcwd())
    os.chdir(wd)
    return(os.listdir())


# # Exploring Data

# In[7]:


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


# ## Looking at variables  as continuous (numeric) , categorical (discrete) and string variables
# -  Note some numeric variables can be categoric. (eg 3,4,8 cylinder cars)
# -  We use values_count.unique to find number of unique values_count for categoric and describe for numeric variables as above in continuous and categorical variales
# -  Categorical values can be made discrete using get_dummy method where each level of categorical variable can be made 1/0 binary variable
# -  Some categoric data can be string
# ###  You can also modify the lists of variables below manually

# In[8]:


# Image(path + "untitled.png")


# In[9]:


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


# # Cleaning Data

# ## Missing Values
# 
# We can do the following with missing values
# #####  -> Drop missing values (way = "drop")
# #####  -> Fill missing values with test statistic (way = "mean" or "median")
# #####  -> impute  missing valus using forward or backfill (way = "ffill" or "bfill" )

# In[10]:


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


# #### -> Impute Missing values using machine learning Algorithm

# In[11]:


# This function uses fancyimpute library to Impute missing values of Numeric Data. The fancy impute uses machine learning algorithms to impute missing values.
def impute_fancy(df, way):#Only valid for numeric Data
    if way == "knn":# If you want to fill missing values using "KNN" algorithm
        df_numeric = df2.select_dtypes(include=[np.float])
        df_filled = pd.DataFrame(KNN(5).complete(df_numeric.as_matrix()), columns= df_numeric.columns, index= df_numeric.index)
    
    elif way == "mice": #If you want to fill missing values using "MICE" algorithm
        df_numeric = df2.select_dtypes(include=[np.float])
        df_filled = pd.DataFrame(mice.complete(df_numeric.as_matrix()), columns= df_numeric.columns, index= df_numeric.index)
    
    return df_filled


# # Data Scaling
# 
# Feature scaling through standardization (or Z-score normalization) can be an important preprocessing step for many machine learning algorithms. Standardization involves rescaling the features such that they have the properties of a standard normal distribution with a mean of zero and a standard deviation of one.

# In[12]:


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


# # Treating Categorical Variables

# 
# - #### Note some numeric variables can be categoric. (eg 3,4,8 cylinder cars)
# - #### We use values_count.unique to find number of unique values_count for categoric and describe for numeric variables as above in continuous and categorical variales
# - #### Categorical values can be made discrete using get_dummy method where each level of categorical variable can be made 1/0 binary variable
# - #### Some categorical data can be string

# # Creating New Features

# In[13]:


#This function creates derived variables and delete older ones which are used to create this. It takes data frame as input.
# data -: Data frame
def new_features(data):
    df['size'] = ['Big' if x >= 4 else 'Small' for x in df['carat']]
    del df["carat"]
    return df


# In[14]:


# This function is used to construct polynomial for all possible combinations upto degree as desired
# data -: Data frame
def poly_features(data):
    poly = PolynomialFeatures(degree=2)# Creating the Object
    X = poly.fit_transform(data)# Transform the data and generate new features
    return X


# In[15]:


# This function takes categorical Columns, Creates Dummy Variable
# Returns a single and final data frame
def dummy_creation(df):
    new_df = pd.get_dummies(df, drop_first= True)
    return new_df


# # Modeling - Process

# In[ ]:


#Modeling Process
# Image("C:\\Users\\ajaohri\\Desktop\\ODSP\\img\\Screenshot 2018-10-18 10.27.15.png")


# # Type of Problem
# ### Categorize by output.
# 
# - If the output of your model is a number, it’s a regression problem.
# - If the output of your model is a class, it’s a classification problem.
# - If the output of your model is a set of input groups, it’s a clustering problem.
# - linear regression is used when the dependent variable is continuous,
# - The predictors can be anything (nominal or ordinal categorical, or continuous, or a mix)
# - You can also convert variables from one type to another
# 
# #### In this case, the output is a Number, so its a Regression problem

# ### Factors affecting the choice of a model are:
# 
# - Whether the model meets the business goals
# - How much pre processing the model needs
# - How accurate the model is
# - How explainable the model is
# - How fast the model is: How long does it take to build a model, and how long does the model take to make predictions.
# - How scalable the model is

# In[ ]:


# Image("C:\\Users\\ajaohri\\Desktop\\ODSP\\img\\ml_map.png") 


# # Splitting the Data into Train and Validation Sets

# In[18]:


#This function takes the input a Dataframe, list of predictor variables and name of response variable. It returns Training and validation set for model
# X-: list of names of predictors
# y-: list of names of response variables
def train_test(df, X, y):
    X_train, X_test, y_train, y_test = train_test_split(df[X], df[y], test_size=0.15)
    return X_train, X_test, y_train, y_test


# # Evalutaion Metrics For Regression

# ### Root Mean Squared Error
# 
# The mean_squared_error function computes mean square error, a risk metric corresponding to the expected value of the squared (quadratic) error or loss.
# 
# ### Mean Absolute Error
# 
# The mean_absolute_error function computes mean absolute error, a risk metric corresponding to the expected value of the absolute error loss or -norm loss.
# 
# ### R2_Score
# 
# The r2_score function computes R², the coefficient of determination. 
# It provides a measure of how well future samples are likely to be predicted by the model. 
# Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). 
# A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0
# 
# 

# In[19]:


# This function calaculates the Root Mean Squared error
def rmse(y_true, y_predict):
    print("The Root Mean Squared Error is: ", np.sqrt(mean_squared_error(y_true, y_predict)))
    return np.sqrt(mean_squared_error(y_true, y_predict))


# In[46]:


#This function calculates the Mean Abolute Error and R2 (Coefficient of determination)
def mape_r2(y_true, y_predict):
    print("The Mean Absolute Error is: ", mean_absolute_error(y_true, y_predict))
    print("The R2_score(Coefficient of Determination) is: ", r2_score(y_true, y_predict))


# # REGRESSION MODELS

# ### Linear Regression
# LinearRegression fits a linear model with coefficients  to minimize the residual sum of squares between the observed responses in the dataset, and the responses predicted by the linear approximation. 
# 
# 
# LinearRegression will take in its fit method arrays X, y and will store the coefficients  of the linear model in its coef_ member:
# 
# coefficient estimates for Ordinary Least Squares rely on the independence of the model terms. When terms are correlated and the columns of the design matrix  have an approximate linear dependence, the design matrix becomes close to singular and as a result, the least-squares estimate becomes highly sensitive to random errors in the observed response, producing a large variance. This situation of multicollinearity can arise, for example, when data are collected without an experimental design.

# ## Polynomial Models
# Generate polynomial and interaction features.
# 
# Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, if an input sample is two dimensional and of the form [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

# ## Ridge , Lasso and ElasticNet Regression
# 
# ### Ridge Regression
# This model solves a regression model where the loss function is the linear least squares function and regularization is given by the l2-norm. Also known as Ridge Regression or Tikhonov regularization.
# 
# Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of coefficients. The ridge coefficients minimize a penalized residual sum of squares. This helps is reducing the overfitting problem
# 
# ### Lasso regression
# Linear Model trained with L1 prior as regularizer (aka the Lasso). The Lasso is a linear model that estimates sparse coefficients. It is useful in some contexts due to its tendency to prefer solutions with fewer parameter values, effectively reducing the number of variables upon which the given solution is dependent. For this reason, the Lasso and its variants are fundamental to the field of compressed sensing.
# 
# Lasso method overcomes the disadvantage of Ridge regression by not only punishing high values of the coefficients β but actually setting them to zero if they are not relevant. Therefore, you might end up with fewer features included in the model than you started with, which is a huge advantage.
# 
# ### Elastic Net Regression
# Linear regression with combined L1 and L2 priors as regularizer. 
# 
# ElasticNet is a linear regression model trained with L1 and L2 prior as regularizer. This combination allows for learning a sparse model where few of the weights are non-zero like Lasso, while still maintaining the regularization properties of Ridge. We control the convex combination of L1 and L2 using the l1_ratio parameter.
# 
# Elastic-net is useful when there are multiple features which are correlated with one another. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.
# 
# A practical advantage of trading-off between Lasso and Ridge is it allows Elastic-Net to inherit some of Ridge’s stability under rotation.

# ### Random Forest Regressor
# 
# A random forest regressor.
# A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default)

# ### XGBoost Regressor
# 
# XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.
# Xgboost or Extreme Gradient Boosting is a very succesful and powerful tree-based algorithm.
# 

# In[21]:


def model_training(X,y, model):
    model.fit(X,y)
    return model


# In[32]:


def model_selection(X_train, y_train, X_test, y_test):
    models = [
        {
    'label': 'Linear Regression',
    'model': LinearRegression(),
        },
        
        {
    'label': 'Ridge Regression',
    'model': Ridge(alpha = 0.4),
        },
        
        {
    'label': 'Lasso Regression',
    'model': Lasso(alpha = 0.1),
        },
        
        {
    'label': 'ElasticNet Regression',
    'model': ElasticNet(l1_ratio= 0.3)
        },
        
        {
    'label': 'RandomForestRegressor',
    'model': RandomForestRegressor(n_estimators= 100, max_depth= 5, max_features=0.8),
        },
        
        {
    'label': 'XGBoost',
    'model': XGBRegressor(n_estimators= 100, max_depth= 5, subsample= 0.7, learning_rate= 0.1),
        }
]
    
    model_fit = []
    
    for m in models:
        model = model_training(X_train, y_train, m["model"])
        print("Model " + m["label"] + " RMSE")
        model_fit.append((model, rmse(y_test, model.predict(X_test))))
        
    print("Different Regression Models with their RMSE Values")
    print(model_fit)
    return  sorted(model_fit, key= lambda x: x[1])[0][0]


# In[44]:


def cal_evaluation(data, y):
    X_train, X_test, y_train, y_test = train_test(data, list(data.drop(y, axis = 1).columns), y)
    final_model = model_selection(X_train, y_train, X_test, y_test)
    print("The Best Model is: ")
    print(final_model)
    y_predict = final_model.predict(X_test)
    print("The RMSE score on Validation Set")
    rmse(y_test, y_predict)
    print("The MAPE and coefficient of determination on Validation set")
    mape_r2(y_test, y_predict)
    return final_model


# # Dump Model to Pickle
# Pickle is a Python libraries which is the best choice to perform the task like
# - Pickling  the process converting any Python object into a stream of bytes by following the hierarchy of the object we are trying to convert. 
# - Unpickling the process of converting the pickled (stream of bytes) back into to the original Python object by following the object hierarchy

# In[37]:


def pickling(name, data, y):
    best_model = cal_evaluation(data, y) #Running all the models and taking the best one
    model_pkl = open(name, 'wb')# Opening the pickle file to dump the model
    pickle.dump(best_model, model_pkl)# Dumping the model in files
    model_pkl.close()# Closing the file


# In[38]:


def model_load(name):
    model_pkl = open(name, 'rb')# Opening thr file in read format
    model = pickle.load(model_pkl)# Loading the pickled model
    print("Loaded model :: ", model)
    return model


# In[42]:


def process(path, csv, y):
    # Reading gthe Data
    df=pd.read_csv(path + csv, index_col= False)
    print(df.head())
    # Removing Unnecessary Columns
    df.drop("Unnamed: 0", axis= 1, inplace= True)
    df = df.drop("measurements", axis=1)
    # Exploring the data
    explore(df)
    #Dropping Rows which have Null values in Response variable
    df = df.dropna(how="any", subset=[y])
    # Exploring the data again
    explore(df)
    # Creating a dictionary which contains names of Categorical and Continuous Variables
    df_cat, df_cont = val_count(df,10)
    # Filling missing values in continuous data
    df_cont = missing_val_treatment_cont(df, df_cont, "median")
    # Scaling the Desired Numeric Columns in Data
    # df_cont = scale_data(df_cont, list(df_cont.columns), "standard")
    #Creating Dummy Variables
    df_cat = dummy_creation(df_cat)
    # Concating both data frames to get final Data
    final_data = pd.concat([df_cont, df_cat], axis=1)
    # Calling the pickle function to dump the best model
    pickling("model.pkl", final_data,y)
    model_load("model.pkl")


# In[45]:

# In[5]:


path='/home/regression/'
set_wd(path)
process(path, "BigDiamonds.csv", "price")


# In[ ]:




