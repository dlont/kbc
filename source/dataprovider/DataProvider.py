import abc
class PandasClassSelector():
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def select(self, df):
        raise NotImplementedError

class PandasSurvivedClassSelector(PandasClassSelector):
    def __init__(self): pass
    def select(self,df):
        return df[df['Survived']==1]
class PandasDrownedClassSelector(PandasClassSelector):
    def __init__(self): pass
    def select(self,df):
        return df[df['Survived']==0]

class SaleMFClassSelector(PandasClassSelector):
    def __init__(self,value=0): self.value = value
    def select(self,df):
        return df[df['Sale_MF']==self.value]
class SaleCCClassSelector(PandasClassSelector):
    def __init__(self,value=0): self.value = value
    def select(self,df):
        return df[df['Sale_CC']==self.value]
class SaleCLClassSelector(PandasClassSelector):
    def __init__(self,value=0): self.value = value
    def select(self,df):
        return df[df['Sale_CL']==self.value]
class SaleRejectedClassSelector(PandasClassSelector):
    def __init__(self): pass
    def select(self,df):
        return df[(df.Sale_MF==0) & (df.Sale_CC==0) & (df.Sale_CL==0)]

class DataProvider():
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def get_training_examples(self,feature_name,selector):
        raise NotImplementedError

    @abc.abstractmethod
    def get_testing_examples(self,feature_name,selector):
        raise NotImplementedError

    @abc.abstractmethod
    def get_training_testing_examples(self,feature_name,selector):
        raise NotImplementedError

    @abc.abstractmethod
    def set_training_testing_splitting(self, fraction):
        '''
        @param fraction: Fraction of examples from the dataset to be used for training
        '''
        raise NotImplementedError

class PandasDataProviderFromCSV(DataProvider):
    def __init__(self,filename_csv):
        import pandas as pd
        from sklearn.model_selection import train_test_split 
        self.filename_csv = filename_csv
        self.training_fraction = 0.5
        self.data = pd.read_csv(self.filename_csv)
        self.train, self.test = train_test_split(self.data, train_size=self.training_fraction)
    
    def get_training_examples(self,feature_name=None,selector=None):
        '''
        Return series of feature values from the training set
        feature_name: optional, feature name of values to be extracted. if None return all features
        selector: optional, selector class to extract subset of examples with specific requirement
                  if None return all examples
        '''
        if not feature_name:
            return self.train
        else:
            if selector: return selector.select(self.train)[feature_name]
            else: return self.train[feature_name]

    def get_testing_examples(self,feature_name=None,selector=None):
        '''
        Return series of feature values from the testing set
        feature_name: optional, feature name of values to be extracted. if None return all features
        selector: optional, selector class to extract subset of examples with specific requirement
                  if None return all examples
        '''
        if not feature_name:
            return self.test
        else:
            if selector: return selector.select(self.test)[feature_name]
            else: return self.test[feature_name]

    def get_training_testing_examples(self,feature_name=None,selector=None):
        '''
        Return series of feature values from the testing set
        feature_name: optional, feature name of values to be extracted. if None return all features
        selector: optional, selector class to extract subset of examples with specific requirement
                  if None return all examples
        '''
        if not feature_name:
            return self.data
        else:
            if selector: return selector.select(self.data)[feature_name]
            else: return self.data[feature_name]

    def set_training_testing_splitting(self, fraction=0.5):
        '''
        @param fraction: Fraction of examples from the dataset to be used for training
        '''
        from sklearn.model_selection import train_test_split
        self.training_fraction = fraction
        self.train, self.test = train_test_split(self.data , train_size=self.training_fraction)

class PandasDataProviderRespondingClientsRevenueMF(PandasDataProviderFromCSV):
        def __init__(self,filename_csv,remove_all=False,training_set=True):
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split 
            # from sklearn.preprocessing import OneHotEncoder
            self.filename_csv = filename_csv
            self.training_fraction = 0.7
            all_data = pd.read_csv(self.filename_csv, index_col='Client')
            self.data = all_data[all_data.Sale_CC != -1]  #training data
            self.data = self.data[self.data.Revenue_MF>0] #select only those client who have non-trivial revenue
            # self.data = all_data[all_data.Sale_CC == -1]    #predictions data
            # transform data using pipelines

            self.train, self.test = train_test_split(self.data, train_size=self.training_fraction, shuffle=False)

class PandasDataProviderRespondingClientsNoOutliers(PandasDataProviderFromCSV):
        def __init__(self,filename_csv,remove_all=False,training_set=True):
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split 
            from sklearn.preprocessing import label_binarize
            # from sklearn.preprocessing import OneHotEncoder
            self.filename_csv = filename_csv
            self.training_fraction = 0.7
            all_data = pd.read_csv(self.filename_csv, index_col='Client')

            #Add multiclass enconded axis. This enconding is bad since it results in overlaping classes. DO NOT USE!

            #Add multiclass binary encoding
            all_data['Sale_Multiclass'] = all_data[['Sale_MF','Sale_CC','Sale_CL']].apply(lambda el:self.ordinal_enconding(el),axis=1)
            multiclass_bin = pd.DataFrame(label_binarize(all_data['Sale_Multiclass'],classes=[0,1,2,3]),index=all_data.index)
            # all_data['Sale_Multiclass'] = all_data[['Sale_MF','Sale_CC','Sale_CL']].apply(lambda el:self.reduced_binary_coding(el),axis=1)
            # multiclass_bin = pd.DataFrame(label_binarize(all_data['Sale_Multiclass'],classes=[0,1,2,3,4]),index=all_data.index)
            # all_data['Sale_Multiclass'] = all_data[['Sale_MF','Sale_CC','Sale_CL']].apply(lambda el:self.complete_binary_coding(el),axis=1)
            # multiclass_bin = pd.DataFrame(label_binarize(all_data['Sale_Multiclass'],classes=[0,1,2,3,4,5,6]),index=all_data.index)
            all_data=pd.merge(all_data,multiclass_bin,left_index=True,right_index=True)

            # all_data['weights'] = all_data['Sale_Multiclass'].apply(self.reduced_binary_encoding_weight)
            
            # print (all_data[['Sale_Multiclass',0,1,2,3,4]].head(20))

            ## resampling majority for balancing
            # from sklearn.utils import resample
            # df_majority = all_data[all_data.Sale_Multiclass==0]
            # df_minority = all_data[all_data.Sale_Multiclass!=0]
            # ## Upsample minority class
            # df_majority_downsampled = resample(df_majority, 
            #                                 replace=False,     # sample with replacement
            #                                 n_samples=150,    # to match majority class
            #                                 random_state=42) # reproducible results
            # all_data = pd.concat([df_minority, df_majority_downsampled])
            # all_data = all_data.sample(frac=1)

            ## Display new class counts
            # print(all_data['Sale_Multiclass'].value_counts())
            # print(all_data.head())

            if training_set: self.data = all_data[all_data.Sale_Multiclass != -1]  #training data
            else: self.data = all_data[all_data.Sale_Multiclass == -1]             #predictions data
            # print (self.data[['Sale_Multiclass',0,1,2,3,4]].head(20))
            
            if remove_all:
                self.data = self.data.drop([27,43,349,614,374,448,479,617,966,1293,1335,1549])                #drop outliers
                self.data = self.data[self.data.ActBal_CA<10000]
            
            self.data['Count_CA']=self.data['Count_CA'].astype('float64')
            self.data['Age']=self.data['Age'].astype('float64')
            self.data['Tenure']=self.data['Tenure'].astype('float64')
            self.data['Sale_Multiclass']=self.data['Sale_Multiclass'].astype('float64')

            # self.data = all_data[all_data.Sale_CC == -1]    #predictions data
            # transform data using pipelines

            self.train, self.test = train_test_split(self.data, train_size=self.training_fraction, shuffle=False)

        def ordinal_enconding(self,el):
            '''
            This is very bad encoding resulting in overlaping classes. DO NOT USE!
            '''
            import numpy as np
            result = -1
            if el['Sale_MF']==1: result = 1
            elif el['Sale_CC']==1: result = 2
            elif el['Sale_CL']==1: result = 3
            elif el['Sale_MF'] == 0 and el['Sale_CC']==0 and el['Sale_CL']==0: result = 0
            elif el['Sale_MF'] == -1 or el['Sale_CC']==-1 or el['Sale_CL']==-1: result = -1
            else: result = np.nan
            return result
            
        def binary_coding(self,el):
            '''
            Improved coding of all possible Sale_MF,Sale_CC,Sale_CL options
            '''
            import numpy as np
            result = -1
            if el['Sale_MF'] == -1 or el['Sale_CC']==-1 or el['Sale_CL']==-1: result = -1
            else: result = el['Sale_MF']*4 + el['Sale_CC']*2 + el['Sale_CL']
            return result


        def complete_binary_coding(self,el):
            '''
            Improved coding of all possible Sale_MF,Sale_CC,Sale_CL options.
            Lumping all cases with two or more Sale_ variables equal into single label.
            Possible lable values are [-1,0,1,2,3,4] or [0,1,2,3,4], when pred class is excluded.
            0 -> Reject
            1 -> Sale_MF=1
            2 -> Sale_CC=1
            3 -> Sale_CL=1
            4 -> Two or more Sale_* variables =1
            '''
            import numpy as np
            result = -1
            if el['Sale_MF'] == -1 or el['Sale_CC']== -1 or el['Sale_CL']== -1: result = -1
            elif el['Sale_MF'] == 0 and el['Sale_CC'] == 0 and el['Sale_CL'] == 0: result = 0
            elif el['Sale_MF'] == 1 and el['Sale_CC'] == 0 and el['Sale_CL'] == 0: result = 1
            elif el['Sale_MF'] == 0 and el['Sale_CC'] == 1 and el['Sale_CL'] == 0: result = 2
            elif el['Sale_MF'] == 0 and el['Sale_CC'] == 0 and el['Sale_CL'] == 1: result = 3
            elif el['Sale_MF'] == 1 and el['Sale_CC'] == 0 and el['Sale_CL'] == 1: result = 4
            elif el['Sale_MF'] == 0 and el['Sale_CC'] == 1 and el['Sale_CL'] == 1: result = 5
            else: result = 6
            return result

        def reduced_binary_coding(self,el):
            '''
            Improved coding of all possible Sale_MF,Sale_CC,Sale_CL options.
            Lumping all cases with two or more Sale_ variables equal into single label.
            Possible lable values are [-1,0,1,2,3,4] or [0,1,2,3,4], when pred class is excluded.
            0 -> Reject
            1 -> Sale_MF=1
            2 -> Sale_CC=1
            3 -> Sale_CL=1
            4 -> Two or more Sale_* variables =1
            '''
            import numpy as np
            result = -1
            if el['Sale_MF'] == -1 or el['Sale_CC']== -1 or el['Sale_CL']== -1: result = -1
            elif el['Sale_MF'] == 0 and el['Sale_CC'] == 0 and el['Sale_CL'] == 0: result = 0
            elif el['Sale_MF'] == 1 and el['Sale_CC'] == 0 and el['Sale_CL'] == 0: result = 1
            elif el['Sale_MF'] == 0 and el['Sale_CC'] == 1 and el['Sale_CL'] == 0: result = 2
            elif el['Sale_MF'] == 0 and el['Sale_CC'] == 0 and el['Sale_CL'] == 1: result = 3
            elif sum([el['Sale_MF'],el['Sale_CC'],el['Sale_CL']])>1: result = 4
            return result

        def reduced_binary_encoding_weight(self,el):
            sum_weights = sum([0.40,0.11,0.14,0.19,0.149])
            my_weights = {-1:1.0,0: 0.40/sum_weights, 1:0.11/sum_weights, 2:0.14/sum_weights, 3:0.19, 4:0.149/sum_weights}
            return my_weights[el]

class PandasDataProviderRespondingClientsNoOutliersRevenueMF(PandasDataProviderFromCSV):
        def __init__(self,filename_csv,remove_all=False,training_set=True):
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split 
            # from sklearn.preprocessing import OneHotEncoder
            self.filename_csv = filename_csv
            self.training_fraction = 0.7
            all_data = pd.read_csv(self.filename_csv, index_col='Client')
            self.data = all_data[all_data.Sale_MF != -1]  #training data
            
            if remove_all:self.data = self.data.drop([27,43,349,614,374,448,479,617,966,1293,1335,1549])                #drop outliers
            else: self.data = self.data.drop([27,43,349,614])                #drop outliers
            
            self.data = self.data[self.data.Revenue_MF>0] #select only those client who have non-trivial revenue
            # self.data = all_data[all_data.Sale_CC == -1]    #predictions data
            # transform data using pipelines

            self.train, self.test = train_test_split(self.data, train_size=self.training_fraction, shuffle=False)

class PandasDataProviderRespondingClientsNoOutliersRevenueCC(PandasDataProviderFromCSV):
        def __init__(self,filename_csv,remove_all=False,training_set=True):
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split 
            # from sklearn.preprocessing import OneHotEncoder
            self.filename_csv = filename_csv
            self.training_fraction = 0.7
            all_data = pd.read_csv(self.filename_csv, index_col='Client')
            self.data = all_data[all_data.Sale_CC != -1]  #training data
            
            if remove_all:self.data = self.data.drop([27,43,349,614,374,448,479,617,966,1293,1335,1549])                #drop outliers
            else: self.data = self.data.drop([374,448,479,617,966,1293,1335])                #drop outliers
            
            self.data = self.data[self.data.Revenue_CC>0] #select only those client who have non-trivial revenue
            # self.data = all_data[all_data.Sale_CC == -1]    #predictions data
            # transform data using pipelines

            self.train, self.test = train_test_split(self.data, train_size=self.training_fraction, shuffle=False)

class PandasDataProviderRespondingClientsNoOutliersRevenueCL(PandasDataProviderFromCSV):
        def __init__(self,filename_csv,remove_all=False,training_set=True):
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split 
            # from sklearn.preprocessing import OneHotEncoder
            self.filename_csv = filename_csv
            self.training_fraction = 0.7
            all_data = pd.read_csv(self.filename_csv, index_col='Client')
            self.data = all_data[all_data.Sale_CL != -1]  #training data
            
            if remove_all:self.data = self.data.drop([27,43,349,614,374,448,479,617,966,1293,1335,1549])                #drop outliers
            else: self.data = self.data.drop([1549])                #drop outliers
            
            self.data = self.data[self.data.Revenue_CL>0] #select only those client who have non-trivial revenue
            # self.data = all_data[all_data.Sale_CC == -1]    #predictions data
            # transform data using pipelines

            self.train, self.test = train_test_split(self.data, train_size=self.training_fraction, shuffle=False)

class PandasDataProviderFromCSV(PandasDataProviderFromCSV):
        def __init__(self,filename_csv):
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split 
            # from sklearn.preprocessing import OneHotEncoder
            self.filename_csv = filename_csv
            self.training_fraction = 0.7
            all_data = pd.read_csv(self.filename_csv, index_col='Client')
            self.data = all_data[all_data.Sale_CC != -1]  #training data
            # self.data = all_data[all_data.Sale_CC == -1]    #predictions data
            # transform data using pipelines

            self.train, self.test = train_test_split(self.data, train_size=self.training_fraction, shuffle=False)

class PandasDataProviderFromCSV_original(PandasDataProviderFromCSV):
        def __init__(self,filename_csv):
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split 
            from sklearn.preprocessing import OneHotEncoder
            self.filename_csv = filename_csv
            self.training_fraction = 0.5
            all_data = pd.read_csv(self.filename_csv, index_col='Client')
            self.data = all_data[all_data.Sale_CC != -1]  #training data
            # self.data = all_data[all_data.Sale_CC == -1]    #predictions data
            # transform data using pipelines

            self.train, self.test = train_test_split(self.data, train_size=self.training_fraction, shuffle=False)