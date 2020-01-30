# TODO

* [ ] **(A)**   30.01.2020 02.45. Incorporate multimodality of `Revenue*` distributions in the models
* [ ] **(A)**   30.01.2020 00.45. Optimize xgboost regression models
  * [ ] **(A)**   30.01.2020 00.45. 'Revenue_MF'
  * [ ] **(A)**   30.01.2020 00.45. 'Revenue_CC'
  * [ ] **(A)**   30.01.2020 00.45. 'Revenue_CL'
* [ ] **(A)**   30.01.2020 00.45. Optimize xgboost classification models
  * [ ] **(A)**   30.01.2020 00.45. 'Sale_MF'
  * [ ] **(A)**   30.01.2020 00.45. 'Sale_CC'
  * [ ] **(A)**   30.01.2020 00.45. 'Sale_CL'
* [ ] **(B)**   28.01.2020 21.20. Ask Michael if missing entries in `Products_ActBalance` can be defaulted to `0` (see [log note](#missing-values-in-products_actbalance))
* [ ] **(B)**   29.01.2020 00.55. Make correlation plots with and without suppression of default imputed values.

* [ ] **(C)**   27.01.2020 23:08. Add automatic provenance generation for the output file. In particular, I wanted to do this for the `train_test_datasets.py` script. It would be nice to have a more general class/function for this purpose. See gdoc "Provenance" section https://docs.google.com/document/d/1YvFl1Dnwc3PGkx154Kpo2dYWy-abask3pxOmDC4Mqo8/edit# for more information
* [ ] **(C1)**  27.01.2020 23:08. I want to have in the provenance metadata file at least the info like: entity, Activity, generatedBy, StartTime, EndTime, md5, git commit.
* [ ] **(C)**   28.01.2020 23:55. Add legend to the correlation plots. At the moment it doesn't look straightforward.
* [ ] **(C)**   28.01.2020 02:26. Check if imputation for gender makes a difference for modelling (see [log note](#gender-missing-values))

---------------------------
# Doing  
  
* [ ] **(A)**   27.01.2020 18:35. Exploratory data analysis
  * [ ] **(A)**   29.01.2020 21.10. Check for outliers in data.
* [ ] **(A)**   28.01.2020 18.15. Build baseline linear and xgboost models.
  * [x] **(A)**   29.01.2020 21.50. Build xgboost baseline for regression 'Revenue_MF','Revenue_CC','Revenue_CL'.
  * [x] **(A)**   29.01.2020 21.50. Build xgboost baseline for classification 'Sale_MF', 'Sale_CC', 'Sale_CL'.
  * [ ] **(A)**   29.01.2020 21.50. Build linear baseline for regression 'Revenue_MF','Revenue_CC','Revenue_CL'.
  * [ ] **(A)**   29.01.2020 21.50. Build svm baseline for classification 'Sale_MF', 'Sale_CC', 'Sale_CL'.
  * [ ] **(B)**   29.01.2020 21.50. Build linear baseline for classification.


-------------------------------
# Follow-up


-----------------------------
# Done 

* [X] **(A1)**  27.01.2020 18:35. Convert multiple files into single .csv file that can be fetched into pandas dataframe
* [X] **(A2)**  27.01.2020 18:35. Plot distributions of all features in all tables
* [X] **(A3)**  28.01.2020 18.00. Plot Sex, Age, Tenure
* [X] **(A4)**  28.01.2020 18.00. Plot 'VolumeCred','VolumeCred_CA','TransactionsCred','TransactionsCred_CA','VolumeDeb','VolumeDeb_CA','VolumeDebCash_Card'
* [X] **(A4)**  28.01.2020 18.00. Plot 'VolumeDebCashless_Card','VolumeDeb_PaymentOrder','TransactionsDeb','TransactionsDeb_CA','TransactionsDebCash_Card','TransactionsDebCashless_Card','TransactionsDeb_PaymentOrder'
* [X] **(A5)**  28.01.2020 18.00. Plot 'Sale_MF','Sale_CC','Sale_CL','Revenue_MF','Revenue_CC','Revenue_CL'
* [X] **(A6)**  28.01.2020 18.00. Plot 'Count_CA','Count_SA','Count_MF','Count_OVD','Count_CC','Count_CL'
* [X] **(A7)**  28.01.2020 18.00. Plot 'ActBal_CA','ActBal_SA','ActBal_MF','ActBal_OVD','ActBal_CC','ActBal_CL'
* [X] **(B)**   27.01.2020 21:43. Ask Michael what is " Inflow/outflow on C/A"? Does C/A stays for current account?
* [X] **(C)**   28.01.2020 00:07. Refactor all occurrences of *`titanic`* symbol in codes
* [X] **(C)**   28.01.2020 00:41. Refactor all `model.build_test_train_*_feature_pad()` methods by moving them to `view` class. This is more logical than having plotting capabilities inside the `model` class
* [X] **(A)**   28.01.2020 12:50. Ask michael about target variables. Are `Sale*` variables categorial. What do they mean? What `Revenue*` variables mean. Should I build a model for functions `Sale(x1,x2,...)` and `Revenue(x1,x2,...)`?

* [X] **(B)**   27.01.2020 18:30. Ask Michael if **Client** is consistent across all tables and can be used as the index for joining
* [X] **(B)**   27.01.2020 20:45. Ask Michael if the tables are randomly reshuffled in order to avoid data leakage in training/testing datasets


----------------------
----------------------

# 27 Jan 2020 18:00
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

# 28 Jan 2020 10:20
## Inflow CA
According to the description table of the .xlsx file, "CA" stays for "current account"

# 28 Jan 2020 10:40
## Missing values

Half of the columns are very well filled. There are only a few missing entries. Perhaps, I can drop those handful of row and continue working with 1587 entries.

```python
df.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1615 entries, 909 to 1466
Data columns (total 35 columns):
Sex                             1612 non-null float64
Age                             1615 non-null int64
Tenure                          1615 non-null int64
VolumeCred                      1587 non-null float64
VolumeCred_CA                   1587 non-null float64
TransactionsCred                1587 non-null float64
TransactionsCred_CA             1587 non-null float64
VolumeDeb                       1587 non-null float64
VolumeDeb_CA                    1587 non-null float64
VolumeDebCash_Card              1587 non-null float64
VolumeDebCashless_Card          1587 non-null float64
VolumeDeb_PaymentOrder          1587 non-null float64
TransactionsDeb                 1587 non-null float64
TransactionsDeb_CA              1587 non-null float64
TransactionsDebCash_Card        1587 non-null float64
TransactionsDebCashless_Card    1587 non-null float64
TransactionsDeb_PaymentOrder    1587 non-null float64
Count_CA                        1615 non-null int64
Count_SA                        426 non-null float64
Count_MF                        306 non-null float64
Count_OVD                       419 non-null float64
Count_CC                        170 non-null float64
Count_CL                        135 non-null float64
ActBal_CA                       1615 non-null float64
ActBal_SA                       426 non-null float64
ActBal_MF                       306 non-null float64
ActBal_OVD                      419 non-null float64
ActBal_CC                       170 non-null float64
ActBal_CL                       135 non-null float64
Sale_MF                         969 non-null float64
Sale_CC                         969 non-null float64
Sale_CL                         969 non-null float64
Revenue_MF                      969 non-null float64
Revenue_CC                      969 non-null float64
Revenue_CL                      969 non-null float64
dtypes: float64(32), int64(3)
memory usage: 454.2 KB
```

The group of columns with 1587 filled entries missing columns seem to be the same. I created a new file `data/28_01_2020_1584entries/data.csv` by removing those rows
```
>>> df[pd.isnull(df['VolumeDeb'])].index.sort_values()
Int64Index([  50,   58,   84,  305,  315,  334,  355,  377,  514,  522,  663,
             723,  741,  799,  833,  941,  962, 1026, 1032, 1139, 1141, 1199,
            1203, 1244, 1269, 1446, 1528, 1544],
           dtype='int64', name=u'Client')
```

# 28 Jan 2020 21:15
## Missing values in Products_ActBalance

Missing entries in the `Products_ActBalance` table seems to correspond to `0` by default, when the product was not used by the client. I created a new file imputing 0, whenever the values in these columns are not available.

```python
import pandas as pd
df = pd.read_csv('data/28_01_2020_1584entries/data_Products_ActBalance_default0.csv', index_col='Client')
df.info()

<class 'pandas.core.frame.DataFrame'>
Int64Index: 1584 entries, 909 to 1466
Data columns (total 35 columns):
Sex                             1584 non-null float64
Age                             1584 non-null int64
Tenure                          1584 non-null int64
VolumeCred                      1584 non-null float64
VolumeCred_CA                   1584 non-null float64
TransactionsCred                1584 non-null float64
TransactionsCred_CA             1584 non-null float64
VolumeDeb                       1584 non-null float64
VolumeDeb_CA                    1584 non-null float64
VolumeDebCash_Card              1584 non-null float64
VolumeDebCashless_Card          1584 non-null float64
VolumeDeb_PaymentOrder          1584 non-null float64
TransactionsDeb                 1584 non-null float64
TransactionsDeb_CA              1584 non-null float64
TransactionsDebCash_Card        1584 non-null float64
TransactionsDebCashless_Card    1584 non-null float64
TransactionsDeb_PaymentOrder    1584 non-null float64
Count_CA                        1584 non-null int64
Count_SA                        1584 non-null float64
Count_MF                        1584 non-null float64
Count_OVD                       1584 non-null float64
Count_CC                        1584 non-null float64
Count_CL                        1584 non-null float64
ActBal_CA                       1584 non-null float64
ActBal_SA                       1584 non-null float64
ActBal_MF                       1584 non-null float64
ActBal_OVD                      1584 non-null float64
ActBal_CC                       1584 non-null float64
ActBal_CL                       1584 non-null float64
Sale_MF                         1584 non-null float64
Sale_CC                         1584 non-null float64
Sale_CL                         1584 non-null float64
Revenue_MF                      1584 non-null float64
Revenue_CC                      1584 non-null float64
Revenue_CL                      1584 non-null float64
dtypes: float64(32), int64(3)

```

# 30.01.2020 02:40
## Multimodal revenue distributions

I guess, baseline modelling is so bad out of the box because the revenues distributions are multimodal with first mode at 0. This mode corresponds to clients that do not want to
accept the marketing offer. Somehow I have to explicitely incorporate this information in the model. Perhaps, I can use gaussian mixture to deal with this problem or find a simpler alternative way.

----------------------------
----------------------------
# Miscellaneous 
## Transformations to the original datasets 

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
