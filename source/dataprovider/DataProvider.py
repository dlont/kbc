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
        def __init__(self,filename_csv):
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
        def __init__(self,filename_csv,remove_all=False):
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split 
            # from sklearn.preprocessing import OneHotEncoder
            self.filename_csv = filename_csv
            self.training_fraction = 0.7
            all_data = pd.read_csv(self.filename_csv, index_col='Client')

            #Add multiclass enconded axis
            all_data['Sale_Multiclass'] = all_data[['Sale_MF','Sale_CC','Sale_CL']].apply(lambda el:self.ordinal_enconding(el),axis=1)

            self.data = all_data[all_data.Sale_Multiclass != -1]  #training data
            
            if remove_all:self.data = self.data.drop([27,43,349,614,374,448,479,617,966,1293,1335,1549])                #drop outliers
            
            # self.data = all_data[all_data.Sale_CC == -1]    #predictions data
            # transform data using pipelines

            self.train, self.test = train_test_split(self.data, train_size=self.training_fraction, shuffle=False)

        def ordinal_enconding(self,el):
            import numpy as np
            result = -1
            if el['Sale_MF']==1: result = 1
            elif el['Sale_CC']==1: result = 2
            elif el['Sale_CL']==1: result = 3
            elif el['Sale_MF'] == 0 and el['Sale_CC']==0 and el['Sale_CL']==0: result = 0
            elif el['Sale_MF'] == -1 or el['Sale_CC']==-1 or el['Sale_CL']==-1: result = -1
            else: result = np.nan
            return result

class PandasDataProviderRespondingClientsNoOutliersRevenueMF(PandasDataProviderFromCSV):
        def __init__(self,filename_csv,remove_all=False):
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
        def __init__(self,filename_csv,remove_all=False):
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
        def __init__(self,filename_csv,remove_all=False):
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