# TODO
  
[] **(B)**   27.01.2020 18:30. Ask Michael if **Client** is consistent across all tables and can be used as the index for joining
[] **(B)**   27.01.2020 20:45. Ask Michael if the tables are randomly reshuffled in order to avoid data leakage in training/testing datasets
[] **(B)**   27.01.2020 21:43. Ask Michael what is " Inflow/outflow on C/A"? Does C/A stays for current account?

[] **(C)**   28.01.2020 00:07. Refactor all occurrences of *titanic* symbol in codes
[] **(C)**   28.01.2020 00:41. Refactor all `model.build_test_train_*_feature_pad()` methods by moving them to `view` class. This is more logical than having plotting capabilities inside the `model` class
[] **(C)**   28.01.2020 02:26. Check if imputation for gender makes a difference for modelling (see [log note](#gender-missing-values))
[] **(C)**   27.01.2020 23:08. Add automatic provenance generation for the output file. In particular, I wanted to do this for the `train_test_datasets.py` script. It would be nice to have a more general class/function for this purpose. See gdoc "Provenance" section https://docs.google.com/document/d/1YvFl1Dnwc3PGkx154Kpo2dYWy-abask3pxOmDC4Mqo8/edit# for more information
[] **(C1)**  27.01.2020 23:08. I want to have in the provenance metadata file at least the info like: entity, Activity, generatedBy, StartTime, EndTime, md5, git commit.

---------------------------
# Doing  
  
[] **(A)**   27.01.2020 18:35. Exploratory data analysis 
[] **(A2)**  27.01.2020 18:35. Plot distributions of all features in all tables

-------------------------------
# Follow-up

-----------------------------
# Done 

[X] **(A1)**  27.01.2020 18:35. Convert multiple files into single .csv file that can be fetched into pandas dataframe

----------------------
----------------------

# 27 Jan 2020 18:00 ->
## Setting up the analysis framework

The data is provided in several tables in .xslx file. It looks like **Client** is a unique ID label shared by all tables. I guess it has to be used to correlated data across different tables. Have to ask Michael.

It would be better to write a script that can be reused if needed to convert individual tables into single .csv file.

# 28 Jan 2020 02:15
## Gender missing values
I found that there are a few missing elements values in the `Sex` column of the `Soc_Dem` table.
```python
import pandas as pd
df = pd.read_csv('data/27_01_2020/data.csv', index_col='Client')
df['Sex'].unique()
> array([ 0.,  1., nan])
df[pd.isnull(df['Sex'])].index
> Int64Index([1363, 953, 843], dtype='int64', name=u'Client')
```
I also checked original files. Indeed, gender is missing for these three entries. Perhaps, I can safely drop these three row, because I have > 1500 entries for the model building, but it is better to compare the results of the training with imputing sex values based on the most frequent occurrence of gender in the dataset (I've made a todo entry for this).

----------------------------
----------------------------
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
