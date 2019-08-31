# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 03:57:22 2019

@author: utsav
"""


import pandas as pd
election = pd.read_csv("C:\\Users\\utsav\\Desktop\\Python_Data_Sets_Data_Camp\\Manipulating DataFrames with pandas\\pennsylvania2012_turnout.csv" ,index_col = "county" , header = 0)

x= 4
y= 4
election.iloc[x ,y] == election.loc['Bedford', 'winner']




results = election[["winner", "total", "voters"]]
results.set_index("winner" , inplace  =True)   

# Print the output of results.head()
print(results.head())


# Slice the row labels 'Potter' to 'Perry' in reverse order: p_counties_rev
p_counties_rev = election.loc["Potter": "Perry": -1]

# Print the p_counties_rev DataFrame
print(p_counties_rev)

# Create the list of row labels: rows
rows = ['Philadelphia', 'Centre', 'Fulton']

# Create the list of column labels: cols
cols = ['winner', 'Obama', 'Romney']

# Create the new DataFrame: three_counties
three_counties = election.loc[rows, cols]

# Print the three_counties DataFrame
print(three_counties)




# Create the boolean array: high_turnout
high_turnout = election["turnout"] > 70
print(high_turnout)
election[high_turnout]
election[election["turnout"] > 70]
# Filter the election DataFrame with the high_turnout array: high_turnout_df
high_turnout_df = election.loc[high_turnout]  

# Print the high_turnout_results DataFrame
print(high_turnout_df)





""" Very Important  Example """
""" Filtering Column based on another """

# Import numpy
import numpy as np
# Create the boolean array: too_close
too_close = election["margin"] < 1

# Assign np.nan to the 'winner' column where the results were too close to call
election["winner"][too_close] = np.nan


print(election[election["margin"]< 1])


print(election.loc[election["margin"] < 1])


print(election.loc[election["margin"] <1]["winner"])


p  = election.loc[election["margin"] < 1]
print(p["winner"])


print(election["winner"][election["margin"] < 1])


""" OR we can use iloc as well """
# Print the output of election.info()
print(election.info())

# Assign np.nan to the 'winner' column where the results were too close to call
election.loc[too_close, 'winner'] = np.nan





import pandas as pd
titanic = pd.read_csv("C:\\Users\\utsav\\Desktop\\Python_Data_Sets_Data_Camp\\Manipulating DataFrames with pandas\\titanic.csv"  , header = 0)


# Select the 'age' and 'cabin' columns: df
df = titanic.loc[:,["age", "cabin"]]
print(df.head())
df_1  = titanic[["age" ,"cabin"]]
print(df_1.head())
# Print the shape of df
print(df.shape)

# Drop rows in df with how='any' and print the shape
print(df.dropna(how = "any" ).shape)

# Drop rows in df with how='all' and print the shape
print(df.dropna(how = "all").shape)

# Drop columns in titanic with less than 1000 non-missing values
print(titanic.dropna(thresh=1000, axis='columns').info())





import pandas as pd
weather = pd.read_csv("C:\\Users\\utsav\\Desktop\\Python_Data_Sets_Data_Camp\\Manipulating DataFrames with pandas\\pittsburgh2013.csv"  , header = 0, index_col =0 ,parse_dates = True)

# Write a function to convert degrees Fahrenheit to degrees Celsius: to_celsius
def to_celsius(F):
    return 5/9*(F - 32)

# Apply the function over 'Mean TemperatureF' and 'Mean Dew PointF': df_celsius

df_celsius = weather[['Mean TemperatureF','Mean Dew PointF']].apply(to_celsius)

# Reassign the columns df_celsius
df_celsius.columns = ['Mean TemperatureC', 'Mean Dew PointC']

# Print the output of df_celsius.head()
print(df_celsius.head())


"map function """
"""The .map() method is used to transform values according to a Python dictionary look-up."""

import pandas as pd
election = pd.read_csv("C:\\Users\\utsav\\Desktop\\Python_Data_Sets_Data_Camp\\Manipulating DataFrames with pandas\\pennsylvania2012_turnout.csv" ,index_col = "county" , header = 0)

red_vs_blue = {"Obama":"blue" , "Romney" :"red"}

election["color"] = election["winner"].map(red_vs_blue)



"""In statistics, the z-score is the number of standard deviations 
by which an observation is above the mean
- so if it is negative, it means the observation is below the mean"""

# Import zscore from scipy.stats
from scipy.stats import zscore

# Call zscore with election['turnout'] as input: turnout_zscore
turnout_zscore =zscore(election["turnout"])

# Print the type of turnout_zscore
print(type(turnout_zscore))

# Assign turnout_zscore to a new column: election['turnout_zscore']
election["turnout_zscore"] = turnout_zscore

# Print the output of election.head()
print(election.head())




"""Chaning the index of a Data Frame """


import pandas as pd
sales = pd.read_csv("C:\\Users\\utsav\\Desktop\\Python_Data_Sets_Data_Camp\\Manipulating DataFrames with pandas\\sales\\sales.csv" )

sales["state"] =["NY" , "NY" , "CA" , "CA" ,"NJ" ,"NJ"]

sales.set_index(["month" ,"state"] ,inplace = True)
print(sales)

#Create a new list of indexes
new_idx = [i.upper() for i in sales.index ]
print(new_idx)

# Assign new_idx to sales.index
sales.index = new_idx
print(sales.index)
print(sales)

 
# Assign the string 'MONTHS' to sales.index.name
sales.index.name = "MONTHS"

# Print the sales DataFrame
print(sales)

# Assign the string 'PRODUCTS' to sales.columns.name 
sales.columns.name = "PRODUCTS"

# Print the sales dataframe again
print(sales)




# Generate the list of months: months
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

# Assign months to sales.index
sales.index = months

# Print the modified sales DataFrame
print(sales)


# Set the index to the column 'state': sales
sales = sales.set_index("state")

# Print the sales DataFrame
print(sales)

# Access the data from 'NY'
print(sales.loc["NY"])



import pandas as pd
sales = pd.read_csv("C:\\Users\\utsav\\Desktop\\Python_Data_Sets_Data_Camp\\Manipulating DataFrames with pandas\\sales\\sales.csv" )

sales["state"] =["NY" , "NY" , "CA" , "CA" ,"NJ" ,"NJ"]


sales["month"] = [1,2,1,2,1,2]


sales.set_index(["state", "month"] ,inplace = True)

print(sales)


print(sales.loc[[("NJ" ,1)] , :])


print(sales.loc[("NJ" ,1) , :])

# Look up data for CA and TX in month 2: CA_TX_month2
CA_TX_month2 = sales.loc[(["CA","NJ"] , 2), : ]
print(CA_TX_month2)


# Look up data for all states in month 2: all_month2
all_month2 = sales.loc[(slice(None),2),:]
print(all_month2)

all_months = sales.loc[("NJ",slice(None)),:]
print(all_months)


all_months_ca_ny = sales.loc[(["CA","NY"],slice(None)),:]
print(all_months)




import pandas as pd
sales = pd.read_csv("C:\\Users\\utsav\\Desktop\\Python_Data_Sets_Data_Camp\\Manipulating DataFrames with pandas\\sales\\sales.csv" , header = 0 , index_col = 0 )

sales["state"] = ["CA","CA","NY","NY","TX","TX"]
sales["month"] = [1,2,1,2,1,2]
print(sales)
sales.set_index(["state","month"] ,inplace = True)
print(sales)



# Print sales.loc[['CA', 'TX']]
print(sales.loc[["CA","TX"]])


# Print sales['CA':'TX']
print(sales["CA" : "TX"])

# Sort the MultiIndex: sales
sales = sales.sort_index(inplace = True)
print(sales)





import pandas as pd
sales = pd.read_csv("C:\\Users\\utsav\\Desktop\\Python_Data_Sets_Data_Camp\\Manipulating DataFrames with pandas\\sales\\sales.csv" , header = 0 , index_col = 0 )
sales["state"] = ["CA","CA","NY","NY","TX","TX"]

sales.set_index("state", inplace = True)
print(sales)

sales.loc[:,"salt"]


# Access the data from 'NY'
print(sales.loc["NY"])



"""Indexing multiple levels of a MultiIndex"""


# Look up data for NY in month 1: NY_month1
NY_month1 = sales.loc[("NY",1), :]

# Look up data for CA and TX in month 2: CA_TX_month2
CA_TX_month2 = sales.loc[(["CA","TX"] , 2), : ]

# Look up data for all states in month 2: all_month2
all_month2 = sales.loc[(slice(None),2),:]
print(all_month2)











import pandas as pd
users = pd.read_csv("C:\\Users\\utsav\\Desktop\\Python_Data_Sets_Data_Camp\\Manipulating DataFrames with pandas\\users.csv" , header = 0 ,index_col =None )
users.rename( columns={'Unnamed: 0':'new column name'}, inplace=True )

users.drop('new column name'  , axis =1 ,inplace = True)
print(users)

# Pivot the users DataFrame: visitors_pivot
visitors_pivot = pd.pivot_table(users, index = "weekday" ,columns = "city" ,values = ["visitors"])
# Print the pivoted DataFrame
print(visitors_pivot)


# Pivot users with signups indexed by weekday and city: signups_pivot
signups_pivot = pd.pivot_table(users , index= "weekday" ,columns = "city", values = ["visitors", "signups"])

# Print signups_pivot
print(signups_pivot)

# Pivot users pivoted by both signups and visitors: pivot
pivot =  pd.pivot_table(users , index= "weekday" ,columns = "city")

# Print the pivoted DataFrame
print(pivot)









""" Unstack and stack the Data in the Data Frame """

"""To build Hierarchical indexes in a Data Frame""" 

import numpy as np
index = pd.MultiIndex.from_tuples([('one', 'a'), ('one', 'b'),
                                    ('two', 'a'), ('two', 'b')])
s = pd.DataFrame({"X" : np.arange(1.0, 5.0) , "Y": np.arange(22,26)}, index=index)

print(s)

s.stack(level = 0)
s.unstack(level = 0)
s.stack(level  = 0)
s.unstack(level=0)

df = s.unstack(level=0)
print(df)
df.stack(level  = 0 )




import pandas as pd
users = pd.read_csv("C:\\Users\\utsav\\Desktop\\Python_Data_Sets_Data_Camp\\Manipulating DataFrames with pandas\\users.csv" , header = 0 ,index_col =None )
users.rename( columns={'Unnamed: 0':'new column name'}, inplace=True )

users.drop('new column name'  , axis =1 ,inplace = True)

users.set_index([ "city" ,"weekday" ], inplace =True)

print(users)
# Unstack users by 'weekday': byweekday
byweekday = users.unstack(level =1)

# Print the byweekday DataFrame
print(byweekday)

# Stack byweekday by 'weekday' and print it
print(byweekday.stack(level ="weekday"))



# Unstack users by 'city': bycity
bycity = users.unstack(level = "city")

# Print the bycity DataFrame
print(bycity)

# Stack bycity by 'city' and print it
print(bycity.stack(level = "city"))



# Stack 'city' back into the index of bycity: newusers
newusers = bycity.stack(level  = "city")

# Swap the levels of the index of newusers: newusers
newusers = newusers.swaplevel(0,1)

# Print newusers and verify that the index is not sorted
print(newusers)

# Sort the index of newusers: newusers
newusers = newusers.sort_index()

# Print newusers and verify that the index is now sorted
print(newusers)

# Verify that the new DataFrame is equal to the original
print(newusers.equals(users))

df_single_level_cols = pd.DataFrame([[0, 1], [2, 3]], 
                                    index=['cat', 'dog'],
                                    columns=['weight', 'height'])
print(df_single_level_cols)
c = df_single_level_cols.stack()
print(c.unstack())





# Set the new index: users_idx
users_idx = users.set_index(["city", "weekday"])

# Print the users_idx DataFrame
print(users_idx)

# Obtain the key-value pairs: kv_pairs
kv_pairs = pd.melt(users_idx, col_level=0)

# Print the key-value pairs
print(kv_pairs)







# Create the DataFrame with the appropriate pivot table: by_city_day
by_city_day = users.pivot_table(index= "weekday" , columns = "city")

# Print by_city_day
print(by_city_day)


# Use a pivot table to display the count of each column: count_by_weekday1
count_by_weekday1 = users.pivot_table(index = "weekday", aggfunc = "count")

# Print count_by_weekday
print(count_by_weekday1)

# Replace 'aggfunc='count'' with 'aggfunc=len': count_by_weekday2
count_by_weekday2 = users.pivot_table(index= "weekday", aggfunc = len)

# Verify that the same result is obtained
print('==========================================')
print(count_by_weekday1.equals(count_by_weekday2))

# Create the DataFrame with the appropriate pivot table: signups_and_visitors
signups_and_visitors = users.pivot_table(index= "weekday" , aggfunc= sum )

# Print signups_and_visitors
print(signups_and_visitors)

# Add in the margins: signups_and_visitors_total 
signups_and_visitors_total = users.pivot_table(index = "weekday", aggfunc = sum , margins= True)

# Print signups_and_visitors_total
print(signups_and_visitors_total)






""" Group By"""


import pandas as pd
titanic = pd.read_csv("C:\\Users\\utsav\\Desktop\\Python_Data_Sets_Data_Camp\\Manipulating DataFrames with pandas\\titanic.csv") 
# Group titanic by 'pclass'
by_class = titanic.groupby("pclass")

# Aggregate 'survived' column of by_class by count
count_by_class = by_class["survived"].count()

""" Or """

titanic_group_by  = titanic.groupby("pclass")["survived"].count()
print(titanic_group_by)

print(count_by_class  == titanic_group_by)

# Print count_by_class
print(count_by_class)



# Group titanic by 'embarked' and 'pclass'
by_mult = titanic.groupby(["embarked", "pclass"])

# Aggregate 'survived' column of by_mult by count
count_mult = by_mult["survived"].count()

# Print count_mult
print(count_mult)


""" Or"""

titanic_group_by_1 = titanic.groupby(["embarked" , "pclass"])["survived"].count()
print(titanic_group_by_1)


print(titanic_group_by_1  == count_mult)



""" Group By Example  """

df = pd.DataFrame({'Animal' : ['Falcon', 'Falcon', 'Parrot', 'Parrot'],
                  'Max Speed' : [380., 370., 24., 26.],
                   "Min Speed" : [11,22,33,44]   })

df.groupby(['Animal']).mean()
df.groupby(df["Animal"]).mean()
df.groupby("Animal")["Min Speed"].mean()



""" Group By Example """

arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot'],
           ['Capitve', 'Wild', 'Capitve', 'Wild']] 

index_of_data_frame = pd.MultiIndex.from_arrays(arrays, names=('Animal', 'Type'))
 
df = pd.DataFrame({'Max Speed' : [390., 350., 30., 20.]},index=index_of_data_frame)
print(df)
df.groupby(level=0).mean()

df.groupby(level  = 1).mean()




""" Group By Example """

gapminder = pd.read_csv("C:\\Users\\utsav\\Desktop\\Python_Data_Sets_Data_Camp\\Manipulating DataFrames with pandas\\gapminder_tidy.csv" ,index_col = "Country") 
print(gapminder.head())

gapminder_group_by = gapminder.groupby(gapminder["region"]).mean()
gapminder.groupby("region").mean()
print(gapminder_group_by.iloc[:,1:])

""" Group By Example Number 2 """

""" Impotant Example """
# Read life_fname into a DataFrame: life
life = pd.read_csv(life_fname, index_col='Country')

# Read regions_fname into a DataFrame: regions
regions = pd.read_csv(regions_fname, index_col = "Country")

# Group life by regions['region']: life_by_region
life_by_region = life.groupby(regions["region"])

# Print the mean over the '2010' column of life_by_region
print(life_by_region["2010"].mean())




""" Group By  Examplpe"""

print(titanic.columns)

a =titanic.groupby("pclass")[["age" , "fare"]].agg(["max", "median"])
print(a)

c = titanic.groupby("pclass")[["age" , "fare" ]].mean()
print(c)

d= titanic.groupby(["pclass", "survived"])["fare"].mean()
print(d)


b =a.loc[:,["age", "max"]]
print(b)


# Group titanic by 'pclass': by_class
by_class = titanic.groupby("pclass")

# Select 'age' and 'fare'
by_class_sub = by_class[['age','fare']]

# Aggregate by_class_sub by 'max' and 'median': aggregated
aggregated = by_class_sub.agg(["max", "median"])
# Print the maximum age in each class
print(aggregated.loc[:,("age" ,"max")])
# Print the median fare in each class
print(aggregated.loc[:,("fare", "median")])






# Read the CSV file into a DataFrame and sort the index: gapminder
gapminder = pd.read_csv('gapminder.csv', index_col=['Year','region','Country']).sort_index()

# Group gapminder by 'Year' and 'region': by_year_region
by_year_region = gapminder.groupby(level=['Year','region'])

# Define the function to compute spread: spread
def spread(series):
    return series.max() - series.min()

# Create the dictionary: aggregator
aggregator = {'population':'sum', 'child_mortality':'mean', 'gdp':spread}

# Aggregate by_year_region using the dictionary: aggregated
aggregated = by_year_region.agg(aggregator)

# Print the last 6 entries of aggregated 
print(aggregated.tail(6))



import pandas as pd
olympics  = pd.read_csv("C:\\Users\\utsav\\Desktop\\Python_Data_Sets_Data_Camp\\Manipulating DataFrames with pandas\\all_medalists.csv" , index_col = "City")

print(olympics.head())


# Select the 'NOC' column of medals: country_names
country_names = olympics["NOC"]
print(country_names.head())

# Count the number of medals won by each country: medal_counts

medal_counts = country_names.value_counts()
print(medal_counts.head())

m  = olympics.groupby("NOC")["Medal"].count()
print(m)


# Construct the pivot table: counted
counted = pd.pivot_table(olympics , index =["NOC"] , columns = ["Medal"] , values = "Athlete" ,aggfunc = "count")
print(counted.head())

# Construct the pivot table: counted
sum_1 = pd.pivot_table(olympics ,index= "NOC" , columns = "Medal" ,  values = "Athlete" , aggfunc = sum)
print(sum_1.head(5))

counted["totals"] = counted.sum(axis = "columns")
print(counted.head())


# Sort counted by the 'totals' column
counted = counted.sort_values("totals",ascending = False)

# Print the top 15 rows of counted
print(counted.head(15))

# Select columns: ev_gen
ev_gen = olympics[["Event_gender", "Gender"]]
print(ev_gen)

# Drop duplicate pairs: ev_gen_uniques
ev_gen_uniques = ev_gen.drop_duplicates()

# Print ev_gen_uniques
print(ev_gen_uniques)


# Group medals by the two columns: medals_by_gender
olympics_by_gender = olympics.groupby(["Event_gender" ,"Gender"])
print(olympics_by_gender)

# Create a DataFrame with a group count: medal_count_by_gender
olympics_count_by_gender = olympics_by_gender.count()

# Print medal_count_by_gender
print(olympics_count_by_gender)



