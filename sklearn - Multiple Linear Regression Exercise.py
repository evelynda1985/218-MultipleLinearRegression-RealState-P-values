#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression with sklearn - Exercise Solution

# You are given a real estate dataset. 
# 
# Real estate is one of those examples that every regression course goes through as it is extremely easy to understand and there is a (almost always) certain causal relationship to be found.
# 
# The data is located in the file: 'real_estate_price_size_year.csv'. 
# 
# You are expected to create a multiple linear regression (similar to the one in the lecture), using the new data. 
# 
# Apart from that, please:
# -  Display the intercept and coefficient(s)
# -  Find the R-squared and Adjusted R-squared
# -  Compare the R-squared and the Adjusted R-squared
# -  Compare the R-squared of this regression and the simple linear regression where only 'size' was used
# -  Using the model make a prediction about an apartment with size 750 sq.ft. from 2009
# -  Find the univariate (or multivariate if you wish - see the article) p-values of the two variables. What can you say about them?
# -  Create a summary table with your findings
# 
# In this exercise, the dependent variable is 'price', while the independent variables are 'size' and 'year'.
# 
# Good luck!

# ## Import the relevant libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbs
sbs.set()

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression


# ## Load the data

# In[3]:


data = pd.read_csv('real_estate_price_size_year.csv')


# In[18]:


data


# In[19]:


data.describe()


# ## Create the regression

# ### Declare the dependent and the independent variables

# In[5]:


x = data[['size','year']]
y = data['price']


# ### Regression

# In[8]:


reg = LinearRegression()
reg.fit(x,y)


# ### Find the intercept

# In[9]:


reg.intercept_


# ### Find the coefficients

# In[10]:


reg.coef_


# ### Calculate the R-squared

# In[11]:


reg.score(x,y)


# ### Calculate the Adjusted R-squared

# In[13]:


x.shape


# In[16]:


r2 = reg.score(x,y)
n = x.shape[0]
p = x.shape[1]

adjusted_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
adjusted_r2


# ### Compare the R-squared and the Adjusted R-squared

# Answer...
# r2 = 0.7764803683276792
# adjusted_r2 = 0.7718717161282499
# It seems the the R-squared is only slightly larger than the Adjusted R-squared, implying that we were not penalized a lot for the inclusion of 2 independent variables.

# ### Compare the Adjusted R-squared with the R-squared of the simple linear regression

# Answer...
# simple linear regression - R-squared:	0.745
# Multiple linear regression adjusted_r2 = 0.7718717161282499
# Comparing the Adjusted R-squared with the R-squared of the simple linear regression (when only 'size' was used - a couple of lectures ago), we realize that 'Year' is not bringing too much value to the result

# ### Making predictions
# 
# Find the predicted price of an apartment that has a size of 750 sq.ft. from 2009.

# In[22]:


reg.predict([[750,2009]])


# ### Calculate the univariate p-values of the variables

# In[23]:


f_regression(x,y)


# In[26]:


p_values = f_regression(x,y)[1]
p_values


# In[27]:


p_values.round(3)


# ### Create a summary table with your findings

# In[31]:


reg_summary = pd.DataFrame(data=x.columns.values, columns=['features'])
reg_summary


# In[34]:


reg_summary['coefficients'] = reg.coef_
reg_summary['p-values'] = p_values.round(3)


# In[35]:


reg_summary


# Answer...
# It seems that 'Year' is not event significant, therefore we should remove it from the model.
