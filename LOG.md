# TODO

[] **(A)**  27.01.2020 18:35. Exploratory data analysis 
[] **(A1)**  27.01.2020 18:35. Convert multiple files into single .csv file that can be fetched into pandas dataframe
[] **(A2)**  27.01.2020 18:35. Plot distributions of all features in all tables
  
[] **(B)**   27.01.2020 18:30. Ask Michael if **Client** is consistent across all tables
  
# Doing  
  
[] (A)   27.01.2020 18:35. Exploratory data analysis 
[] (A1)  27.01.2020 18:35. Convert multiple files into single .csv file that can be fetched into pandas dataframe
[] 
# Follow-up

# Done 

# 27 Jan 2020 18:00
## Setting up the analysis framework

The data is provided in several tables in .xslx file. It looks like **Client** is a unique ID label shared by all tables. I guess it has to be used to correlated data across different tables. Have to ask Michael.

It would be better to write a script that can be reused if needed to convert individual tables into single .csv file.

# Titanic (old)
## Trasnformations to the original datasets 

```python
# Modify train.csv, test.csv

# python

import pandas as pd
import numpy as np
df = pd.read_csv('data/original/test.csv', index_col='PassengerId')
df['Sex'] = df['Sex'].replace({'male':0,'female':1})
df['Name']=df['Name'].apply(lambda en: en.split(',')[0])
#object_cols = ['Embarked']
df['Embarked'] = df['Embarked'].replace({np.nan:0,'S':1,'C':2,'Q':3})
df.to_csv('data/2019-12-05/test.csv')
```
