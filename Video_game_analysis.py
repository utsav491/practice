# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 05:44:53 2020

@author: utsav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("E:\\Practice\\vgsales.csv\\vgsales.csv")
df = df.set_index('Rank')
print(df)
df.head()
df.columns
df.info()
df.isnull().sum()
df.isnull().sum().plot(kind = 'bar', color = "green")
df["Year"].dtype


df["Publisher"].dtype
df["Publisher"].value_counts()

#df["NA_Sales"] = df["NA_Sales"].astype('object')
#df["NA_Sales"] = df["NA_Sales"].apply(lambda x: x + "$")

mean = df["EU_Sales"].mean()
df["EU_Sales"] = df["EU_Sales"].apply(lambda y : y- mean)

df["E"]  = df.loc[:,["NA_Sales","EU_Sales"]].sum(axis =1)
df["f"] = df.apply(lambda s: int(s.NA_Sales) - int(s.EU_Sales), axis= 1)
df["g"] = df.apply(lambda w : w.JP_Sales - w.Other_Sales, axis  = 1)
df["g"]

df["Platform"].value_counts().plot(kind ="bar", color = "black")
import seaborn as sns
b = df.groupby("Year").sum()[["JP_Sales","Other_Sales"]]
print(b)
"""Platform Sales"""
sns.set(style='darkgrid')

df["Other_Sales"] = df["Other_Sales"] * 10
df.groupby(["Platform"]).mean()["Other_Sales"].plot()
.plot(kind = "bar", color = "green")

a = df["Year"]
sns.set(style='darkgrid')
sns.lineplot(x=a, y=b.iloc[:,0], data= df)
sns.lineplot(x = a, y = b.iloc[:,1],  data = df)
sales_percontinent_per_plaform = df.groupby("Platform").sum()[["NA_Sales"]]
print(sales_percontih  nent_per_plaform)


plt.figure(figsize=(20,10)) 
sales_percontinent_per_plaform.plot(kind = 'bar')
plt.show()

df["Genre"].value_counts()

continents  = list(df.columns[-4:])
print(continents)
def sum_sale_per_continent_per_platform(groupby_element, continent):
    sales_percontinent_per_plaform = df.groupby(groupby_element).sum()[[continent]]
    sales_percontinent_per_plaform.plot(kind = 'bar')
    plt.show()

    sum_sale_per_continent_per_platform(continents,"Platform")


for a in continents:
    sum_sale_per_continent_per_platform(a,"Platform")
for b in genre:
    sum_sale_per_continent_per_platform(b)
    
    df.groupby
    
    
    
    
    