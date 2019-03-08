import numpy as np
import pandas as pd

df = pd.read_csv("C:\\Users\\utsav\\Desktop\\Python_Data_Sets_Data_Camp\\Merging Data Frames in Pandas\\Sales\\sales-jan-2015.csv" , index_col = "Date" , parse_dates= True))
print(df.head())
