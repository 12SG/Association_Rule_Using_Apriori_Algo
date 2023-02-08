#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


pip install apyori


# In[3]:


from apyori import apriori


# In[4]:


data = pd.read_csv("C:/Users/HP/Downloads/Groceries_dataset.csv.zip")


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


##Checking availability of NULL values
data.isnull().sum().sort_values(ascending=False)


# ### Data Preprocessing

# In[8]:


## Type Conversion from Object to Datetime
data['Date'] = pd.to_datetime(data['Date'])
data.info()


# In[9]:


data.head()


# ### Performing EDA 

# In[10]:


## Creating distribution of Item Sold

Item_distr = data.groupby(by = "itemDescription").size().reset_index(name='Frequency').sort_values(by = 'Frequency',ascending=False).head(10)


# In[11]:


#Declearing variable
bars = Item_distr["itemDescription"]
height = Item_distr["Frequency"]
x_pos = np.arange(len(bars))


# In[12]:


#Figsize
plt.figure(figsize=(16,9))


# In[13]:


#Create Bar
plt.bar(x_pos, height, color=(0.3, 0.4, 0.6, 0.6))

# Add title and axis names
plt.title("Top 10 Sold Items")
plt.xlabel("Item Name")
plt.ylabel("Number of Quantity Sold")

# Create names on the x-axis
plt.xticks(x_pos, bars)

# Show graph
plt.show()


# ### Month Year Sale 

# In[14]:


data_date=data.set_index(['Date']) ## Setting date as index for plotting purpose
data_date


# In[15]:


data_date.resample("M")['itemDescription'].count().plot(figsize = (20,8), grid = True, title = "Number by Items Sold by Month").set(xlabel = "Date", ylabel = "Number of Items Sold")


# ### Apriori Implementation

# In[16]:


#Data Preparation
cust_level = data[["Member_number", "itemDescription"]].sort_values(by = "Member_number", ascending = False)
## Selecting only required variables for modelling
cust_level['itemDescription'] = cust_level['itemDescription'].str.strip()
# Removing white spaces if any
cust_level


# In[17]:


#Create Transiction List
transactions = [a[1]['itemDescription'].tolist() for a in list(cust_level.groupby(['Member_number']))]
## Combing all the items in list format for each cutomer


# ### Train Model 

# In[18]:



rules = apriori(transactions = transactions, min_support = 0.002, min_confidence = 0.05, min_lift = 3, min_length = 2, max_length = 2) 


# In[19]:


results = list(rules)


# In[20]:


results


# ### Result Customization 

# In[21]:


## Creating user-defined function for arranging the results obtained from model into readable format

def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])


# In[22]:


resultsinDataFrame.nlargest(n=10, columns="Lift") 


# In[ ]:




