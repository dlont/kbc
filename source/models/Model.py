import pandas as pd
import numpy as np

from logwith import *

class Model(object):
        def __init__(self,configuration):
                self._configuration = configuration
                self._objects = {}
                self._annotation = 'Performance comparision of different MVA discriminants'
                if 'annotation' in self._configuration:
                        self._annotation = self._configuration['annotation']
                self.Initialize()

        @log_with()
        def Initialize(self):
                pass

        @log_with()
        def get(self,name):
                """
                Factory method
                """
                if name in self._objects:
                        return self._objects[name]
                else:
                        return None #provide factory method implementation here
                return self._objects[name]

        @log_with()
        def predict_kaggle_output(ml_model):
                pass

class AdvancedModel(Model):
        def __init__(self,configuration):
                self._configuration = configuration
                self._objects = {}
                self._annotation = 'Performance comparision of different MVA discriminants'
                if 'annotation' in self._configuration:
                        self._annotation = self._configuration['annotation']
                self.Initialize()

        @log_with()
        def Initialize(self):
                pass

        @log_with()
        def get(self,name):
                """
                Factory method
                """
                if name in self._objects:
                        return self._objects[name]
                else:
                        return None #provide factory method implementation here
                return self._objects[name]

        @log_with()
        def get_data_provider(self,name):
                """
                Factory method for data providers
                """
                from dataprovider import PandasDataProviderFromCSV
                if name in self._objects:
                        return self._objects[name]
                else:
                        if '.csv' in self._configuration[name]['input_file']:
                                provider = PandasDataProviderFromCSV(self._configuration[name]['input_file'])
                                self._objects[name] = provider
                        else: raise NotImplementedError
                return self._objects[name]
  
        @log_with()
        def build_best_prediction(self):
                print "Dummy building advanced model!"
                # from xgboost import XGBClassifier
                # from sklearn.metrics import classification_report
                # from dataprovider import PandasSurvivedClassSelector, PandasDrownedClassSelector
                
                # data_provider = self.get_data_provider(self._configuration['Age']['data_provider'])

                # my_model = XGBClassifier()
                # X_train, y_train = data_provider.train.drop(['Cabin','Ticket','Name','Survived'],axis=1),data_provider.train['Survived']
                # # print X_train.dtypes

                # my_model.fit(X_train, y_train)

                # X_test, y_test = data_provider.test.drop(['Cabin','Ticket','Name','Survived'],axis=1),data_provider.test['Survived']
                # y_pred = my_model.predict(X_test)
                # print classification_report(y_test, y_pred, target_names=['Survived','Drowned'])

                # self.predict_kaggle_output(my_model)

                pass

        @log_with()
        def predict_kaggle_output(self,ml_model):
                from sklearn.preprocessing import OneHotEncoder
                kaggle_input = pd.read_csv('data/2019-12-05/test.csv', index_col='Client')
                # transform data using pipelines
                kaggle_input = kaggle_input.astype({'Age':'float32','Sex':'int32'})

                # Apply one-hot encoder to each column with categorical data
                object_cols = ['Embarked']
                data_provider = self.get_data_provider(self._configuration['Age']['data_provider'])
                OH_encoder = data_provider.OH_encoder
                OH_cols_data = pd.DataFrame(OH_encoder.transform(kaggle_input[object_cols]))
                OH_cols_data = OH_cols_data.rename(columns={0:"Embarked_Unknw",1:"Embarked_S",2:"Embarked_C",3:"Embarked_Q"})
                OH_cols_data.index = kaggle_input.index
                # num_data = self.data.drop(object_cols, axis=1)
                kaggle_input = pd.concat([kaggle_input,OH_cols_data], axis=1)

                X_pred = kaggle_input.drop(['Cabin','Ticket','Name'],axis=1)
                y_pred = ml_model.predict(X_pred)
                kaggle_prediction = pd.DataFrame(y_pred,columns=['Survived'],index=X_pred.index)
                kaggle_prediction.to_csv('results/2.3pre.csv')
                pass


class VanillaModel(Model):
        def __init__(self,configuration):
                self._configuration = configuration
                self._objects = {}
                self._annotation = 'Performance comparision of different MVA discriminants'
                if 'annotation' in self._configuration:
                        self._annotation = self._configuration['annotation']
                self.Initialize()

        @log_with()
        def Initialize(self):
                pass

        @log_with()
        def get(self,name):
                """
                Factory method
                """
                if name in self._objects:
                        return self._objects[name]
                else:
                        return None #provide factory method implementation here
                return self._objects[name]

        @log_with()
        def get_data_provider(self,name):
                """
                Factory method for data providers
                """
                from dataprovider import PandasDataProviderFromCSV_original
                if name in self._objects:
                        return self._objects[name]
                else:
                        if '.csv' in self._configuration[name]['input_file']:
                                provider = PandasDataProviderFromCSV_original(self._configuration[name]['input_file'])
                                self._objects[name] = provider
                        else: raise NotImplementedError
                return self._objects[name]

        @log_with()
        def build_best_prediction(self):
                print "Dummy building vanilla model!"

                from xgboost import XGBRegressor
                # from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error
                from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
                
                data_provider = self.get_data_provider(self._configuration['Sex']['data_provider'])

                my_model = XGBRegressor()
                X_train = data_provider.train.drop(['Sale_MF','Sale_CC','Sale_CL','Revenue_MF','Revenue_CC','Revenue_CL'],axis=1)
                # y_train = data_provider.train[['Sale_MF','Sale_CC','Sale_CL','Revenue_MF','Revenue_CC','Revenue_CL']]
                y_train = data_provider.train['Revenue_MF']
                
                print X_train.dtypes
                print X_train.head()

                print y_train.dtypes
                print y_train.head()

                my_model.fit(X_train, y_train)

                X_true = data_provider.test.drop(['Sale_MF','Sale_CC','Sale_CL','Revenue_MF','Revenue_CC','Revenue_CL'],axis=1)
                y_true = data_provider.test['Revenue_MF']
                y_pred = my_model.predict(X_true)
                # print "Max error: ", max_error(y_true,y_pred)
                print "Explained variance score: ", explained_variance_score(y_true,y_pred)
                print "Mean absolute error: ", mean_absolute_error(y_true,y_pred)
                print "Mean squared error: ", mean_squared_error(y_true,y_pred)

                pass

