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

class DataProvider():
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def get_training_examples(self,feature_name,selector):
        raise NotImplementedError

    @abc.abstractmethod
    def get_training_examples(self,feature_name,selector):
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

    def set_training_testing_splitting(self, fraction=0.5):
        '''
        @param fraction: Fraction of examples from the dataset to be used for training
        '''
        from sklearn.model_selection import train_test_split
        self.training_fraction = fraction
        self.train, self.test = train_test_split(self.data , train_size=self.training_fraction)

class PandasDataProviderFromCSV(PandasDataProviderFromCSV):
        def __init__(self,filename_csv):
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split 
            from sklearn.preprocessing import OneHotEncoder
            self.filename_csv = filename_csv
            self.training_fraction = 0.5
            self.data = pd.read_csv(self.filename_csv, index_col='Client')
            # transform data using pipelines
            for col in ['Sale_MF','Sale_CC','Sale_CL','Revenue_MF','Revenue_CC','Revenue_CL']: 
                self.data[col] = self.data[col].fillna(-1)

            self.train, self.test = train_test_split(self.data, train_size=self.training_fraction)

class PandasDataProviderFromCSV_original(PandasDataProviderFromCSV):
        def __init__(self,filename_csv):
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split 
            from sklearn.preprocessing import OneHotEncoder
            self.filename_csv = filename_csv
            self.training_fraction = 0.5
            self.data = pd.read_csv(self.filename_csv, index_col='Client')
            # transform data using pipelines

            self.train, self.test = train_test_split(self.data, train_size=self.training_fraction)