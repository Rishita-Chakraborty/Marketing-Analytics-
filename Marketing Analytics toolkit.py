#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

# Generate random data for clusters
np.random.seed(42)
cluster_1 = np.random.normal(loc=[25, 30], scale=3, size=(30,2))
cluster_2 = np.random.normal(loc=[40, 70], scale=3, size=(30,2))
cluster_3 = np.random.normal(loc=[60, 40], scale=3, size=(30,2))

data = np.vstack((cluster_1, cluster_2, cluster_3))
labels = np.array([0]*30 + [1]*30 + [2]*30)

# Scatter plot
plt.figure(figsize=(6,4))
plt.scatter(data[:,0], data[:,1], c=labels, cmap='viridis', s=40)
plt.title('Customer Segments (K-Means Clustering)')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.grid(True)
plt.tight_layout()
plt.savefig('cluster_analysis.png')
plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# Price range
price = np.linspace(100, 500, 100)
# Logistic function for probability
probability = 1 / (1 + np.exp(0.02*(price-250)))

plt.figure(figsize=(6,4))
plt.plot(price, probability, color='blue')
plt.title('Probability of Purchase vs. Price')
plt.xlabel('Price ($)')
plt.ylabel('Purchase Probability')
plt.grid(True)
plt.tight_layout()
plt.savefig('choice_model.png')
plt.show()


# In[3]:


import matplotlib.pyplot as plt

# Attributes and utilities
attributes = ['Battery Life', 'Price', 'Brand', 'Camera Quality']
utilities = [0.8, 0.5, 0.3, 0.4]

plt.figure(figsize=(6,4))
plt.bar(attributes, utilities, color='skyblue')
plt.title('Part-Worth Utilities for Smartphone Attributes')
plt.ylabel('Utility Score')
plt.tight_layout()
plt.savefig('conjoint_analysis.png')
plt.show()


# In[4]:


import numpy as np
import matplotlib.pyplot as plt

# Time steps (weeks)
weeks = np.arange(0, 10)
# Adstock decay
lambda_decay = 0.8
adstock = lambda_decay**weeks

plt.figure(figsize=(6,4))
plt.plot(weeks, adstock, marker='o', color='green')
plt.title('Adstock Decay Curve (Advertising Memory)')
plt.xlabel('Weeks')
plt.ylabel('Adstock Effect')
plt.grid(True)
plt.tight_layout()
plt.savefig('adstock_decay.png')
plt.show()


# In[ ]:




