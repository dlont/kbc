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

        # @log_with()
        # def predict_kaggle_output(self,ml_model):
        #         from sklearn.preprocessing import OneHotEncoder
        #         kaggle_input = pd.read_csv('data/2019-12-05/test.csv', index_col='Client')
        #         # transform data using pipelines
        #         kaggle_input = kaggle_input.astype({'Age':'float32','Sex':'int32'})

        #         # Apply one-hot encoder to each column with categorical data
        #         object_cols = ['Embarked']
        #         data_provider = self.get_data_provider(self._configuration['Age']['data_provider'])
        #         OH_encoder = data_provider.OH_encoder
        #         OH_cols_data = pd.DataFrame(OH_encoder.transform(kaggle_input[object_cols]))
        #         OH_cols_data = OH_cols_data.rename(columns={0:"Embarked_Unknw",1:"Embarked_S",2:"Embarked_C",3:"Embarked_Q"})
        #         OH_cols_data.index = kaggle_input.index
        #         # num_data = self.data.drop(object_cols, axis=1)
        #         kaggle_input = pd.concat([kaggle_input,OH_cols_data], axis=1)

        #         X_pred = kaggle_input.drop(['Cabin','Ticket','Name'],axis=1)
        #         y_pred = ml_model.predict(X_pred)
        #         kaggle_prediction = pd.DataFrame(y_pred,columns=['Survived'],index=X_pred.index)
        #         kaggle_prediction.to_csv('results/2.3pre.csv')
        #         pass

class AdvancedModelRegression(Model):
        def __init__(self,configuration):
                self._configuration = configuration
                self._objects = {}
                self._annotation = 'Performance comparision of different MVA discriminants'
                if 'annotation' in self._configuration:
                        self._annotation = self._configuration['annotation']
                self.fit_results = None
                self.Initialize()

        @log_with()
        def Initialize(self):
                self.build_best_prediction()
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
                from dataprovider import *
                if name in self._objects:
                        return self._objects[name]
                else:
                        if '.csv' in self._configuration[name]['input_file']:
                                if self._configuration[name]['type'] =='PandasDataProviderRespondingClientsRevenueMF': 
                                        provider = PandasDataProviderRespondingClientsRevenueMF(self._configuration[name]['input_file'])
                                elif self._configuration[name]['type'] =='PandasDataProviderRespondingClientsNoOutliersRevenueMF': 
                                        provider = PandasDataProviderRespondingClientsNoOutliersRevenueMF(self._configuration[name]['input_file'])
                                else: raise NotImplementedError
                                self._objects[name] = provider
                        else: raise NotImplementedError
                return self._objects[name]

        @log_with()
        def build_best_prediction(self):
                print "Dummy building vanilla model!"

                from matplotlib import pyplot
                from xgboost import XGBRegressor, plot_importance
                # from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error
                from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
                
                target_variable_names = self._configuration['model']['target'][0]
                data_provider = self.get_data_provider(self._configuration['model']['data_provider'])

                input_features_names = self._configuration['model']['input_features']
                X_train = data_provider.train[input_features_names]
                y_train = data_provider.train[target_variable_names]

                X_test = data_provider.test[input_features_names]
                y_test = data_provider.test[target_variable_names]

                # print X_train.dtypes
                # print X_train.head()
                # print X_test.dtypes
                # print X_test.head()

                # print y_train.dtypes
                # print y_train.head()
                # print y_test.dtypes
                # print y_test.head()

                eval_set = [(X_train, y_train), (X_test, y_test)]

                my_model = XGBRegressor(n_estimators=self._configuration['model']['n_estimators'],
                                        max_depth=self._configuration['model']['max_depth'],
                                        learning_rate=self._configuration['model']['learning_rate'],
                                        verbosity=0)
                my_model.fit(X_train, y_train, eval_metric=["rmse", "mae"], eval_set=eval_set, verbose=False)

                y_pred = my_model.predict(X_test)
                # print "Max error: ", max_error(y_test,y_pred)
                print "Explained variance score: ", explained_variance_score(y_test,y_pred)
                print "Mean absolute error: ", mean_absolute_error(y_test,y_pred)
                print "Mean squared error: ", mean_squared_error(y_test,y_pred)

                self.fit_results = my_model.evals_result()
                # print 'YO importance'
                # plot_importance(my_model)

                pass

class VanillaModelRegression(Model):
        def __init__(self,configuration):
                self._configuration = configuration
                self._objects = {}
                self._annotation = 'Performance comparision of different MVA discriminants'
                if 'annotation' in self._configuration:
                        self._annotation = self._configuration['annotation']
                self.fit_results = None
                self.Initialize()

        @log_with()
        def Initialize(self):
                self.build_best_prediction()
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

                from matplotlib import pyplot
                from xgboost import XGBRegressor, plot_importance
                # from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error
                from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
                
                target_variable_names = self._configuration['model']['target'][0]
                data_provider = self.get_data_provider(self._configuration['model']['data_provider'])

                input_features_names = self._configuration['model']['input_features']
                X_train = data_provider.train[input_features_names]
                y_train = data_provider.train[target_variable_names]

                X_test = data_provider.test[input_features_names]
                y_test = data_provider.test[target_variable_names]

                # print X_train.dtypes
                # print X_train.head()
                # print X_test.dtypes
                # print X_test.head()

                # print y_train.dtypes
                # print y_train.head()
                # print y_test.dtypes
                # print y_test.head()

                eval_set = [(X_train, y_train), (X_test, y_test)]

                my_model = XGBRegressor(n_estimators=self._configuration['model']['n_estimators'],
                                        max_depth=self._configuration['model']['max_depth'],
                                        learning_rate=self._configuration['model']['learning_rate'],
                                        verbosity=0)
                my_model.fit(X_train, y_train, eval_metric=["rmse", "mae"], eval_set=eval_set, verbose=False)

                y_pred = my_model.predict(X_test)
                # print "Max error: ", max_error(y_test,y_pred)
                print "Explained variance score: ", explained_variance_score(y_test,y_pred)
                print "Mean absolute error: ", mean_absolute_error(y_test,y_pred)
                print "Mean squared error: ", mean_squared_error(y_test,y_pred)

                self.fit_results = my_model.evals_result()
                # print 'YO importance'
                # plot_importance(my_model)

                pass

class VanillaModelLassoLarsIC(Model):
        def __init__(self,configuration):
                self._configuration = configuration
                self._objects = {}
                self._annotation = 'Ridge linear regression with built-in CV'
                if 'annotation' in self._configuration:
                        self._annotation = self._configuration['annotation']
                self.fit_results = None
                self.Initialize()

        @log_with()
        def Initialize(self):
                self.build_best_prediction()
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
                print "Building LassoLarsIC linear regression vanilla model!"

                from matplotlib import pyplot
                from sklearn.linear_model import LassoLarsIC
                from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
                
                target_variable_names = self._configuration['model']['target'][0]
                data_provider = self.get_data_provider(self._configuration[target_variable_names]['data_provider'])

                input_features_names = self._configuration['model']['input_features']
                X_train = data_provider.train[input_features_names]
                y_train = data_provider.train[target_variable_names]

                X_test = data_provider.test[input_features_names]
                y_test = data_provider.test[target_variable_names]

                # print X_train.dtypes
                # print X_train.head()
                # print X_test.dtypes
                # print X_test.head()

                # print y_train.dtypes
                # print y_train.head()
                # print y_test.dtypes
                # print y_test.head()

                my_model_aic = LassoLarsIC(criterion='aic')
                my_model_aic.fit(X_train, y_train)
                y_pred_aic = my_model_aic.predict(X_test)
                # print "Max error: ", max_error(y_test,y_pred)
                print "AIC Explained variance score: ", explained_variance_score(y_test,y_pred_aic)
                print "AIC Mean absolute error: ", mean_absolute_error(y_test,y_pred_aic)
                print "AIC Mean squared error: ", mean_squared_error(y_test,y_pred_aic)

                my_model_bic = LassoLarsIC(criterion='bic')
                my_model_bic.fit(X_train, y_train)
                y_pred_bic = my_model_bic.predict(X_test)
                # print "Max error: ", max_error(y_test,y_pred)
                print "BIC Explained variance score: ", explained_variance_score(y_test,y_pred_bic)
                print "BIC Mean absolute error: ", mean_absolute_error(y_test,y_pred_bic)
                print "BIC Mean squared error: ", mean_squared_error(y_test,y_pred_bic)

                self.fit_results = {'aic':my_model_aic, 'bic':my_model_bic}

                pass
class VanillaModelClassification(Model):
        def __init__(self,configuration):
                self._configuration = configuration
                self._objects = {}
                self._annotation = 'Performance comparision of different MVA discriminants'
                if 'annotation' in self._configuration:
                        self._annotation = self._configuration['annotation']
                self.fit_results = None
                self.Initialize()

        @log_with()
        def Initialize(self):
                self.build_best_prediction()
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

                from xgboost import XGBClassifier
                # from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error
                from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
                
                target_variable_names = self._configuration['model']['target'][0]
                data_provider = self.get_data_provider(self._configuration[target_variable_names]['data_provider'])

                input_features_names = self._configuration['model']['input_features']
                X_train = data_provider.train[input_features_names]
                y_train = data_provider.train[target_variable_names]

                X_test = data_provider.test[input_features_names]
                y_test = data_provider.test[target_variable_names]

                # print X_train.dtypes
                # print X_train.head()
                # print X_test.dtypes
                # print X_test.head()

                # print y_train.dtypes
                # print y_train.head()
                # print y_test.dtypes
                # print y_test.head()

                eval_set = [(X_train, y_train), (X_test, y_test)]

                my_model = XGBClassifier(n_estimators=self._configuration['model']['n_estimators'],
                                        max_depth=self._configuration['model']['max_depth'],
                                        learning_rate=self._configuration['model']['learning_rate'],
                                        verbosity=0)
                my_model.fit(X_train, y_train, eval_metric=["error", "auc", "map"], eval_set=eval_set, verbose=False)

                # y_pred = my_model.predict(X_test)
                # print "Max error: ", max_error(y_test,y_pred)
                # print "Explained variance score: ", explained_variance_score(y_test,y_pred)
                # print "Mean absolute error: ", mean_absolute_error(y_test,y_pred)
                # print "Mean squared error: ", mean_squared_error(y_test,y_pred)

                self.fit_results = my_model.evals_result()
                # print 'YO importance'
                # plot_importance(my_model)

                pass

