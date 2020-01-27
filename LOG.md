# 27 Jan 2020 18:00
## Setting up the analysis framework

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
