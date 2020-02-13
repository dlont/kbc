import pandas as pd
import numpy as np

import pickle
import time

from logwith import *

from dataprovider import *

# Utility function to report best scores
def report(results, n_top=3):
        for i in range(1, n_top + 1):
                candidates = np.flatnonzero(results['rank_test_score'] == i)
                for candidate in candidates:
                        print("Model with rank: {0}".format(i))
                        print("Mean validation score: {0:.3f} (std: {1:.3f})"
                                .format(results['mean_test_score'][candidate],
                                        results['std_test_score'][candidate]))
                        print("Mean training score: {0:.3f} (std: {1:.3f})"
                                .format(results['mean_train_score'][candidate],
                                        results['std_train_score'][candidate]))
                        print("Parameters: {0}".format(results['params'][candidate]))
                        print("")

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

class AdvancedModelClassificationOvRRF(Model):
        def __init__(self,configuration):
                self._configuration = configuration
                self._objects = {}
                self._annotation = 'One VS Rest multiclass Random Forest model'
                if 'annotation' in self._configuration:
                        self._annotation = self._configuration['annotation']
                self.do_training = self._configuration['model'].get('do_training',False)
                self.fit_results = None
                self.my_model = None
                self.Initialize()

        @log_with()
        def Initialize(self):
                if self.do_training: self.build_best_prediction()
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
        def get_data_provider(self,provider_name):
                """
                Factory method for data providers
                """
                if provider_name in self._objects:
                        return self._objects[provider_name]
                else:
                        if '.csv' in self._configuration[provider_name]['input_file']:
                                if self._configuration[provider_name]['type'] =='PandasDataProviderRespondingClients': 
                                        raise NotImplementedError
                                elif self._configuration[provider_name]['type'] =='PandasDataProviderRespondingClientsNoOutliers': 
                                        provider = PandasDataProviderRespondingClientsNoOutliers(self._configuration[provider_name]['input_file'],
                                        remove_all=self._configuration[provider_name]['remove_all'],training_set=self._configuration[provider_name]['training_set'])
                                else: raise NotImplementedError
                                self._objects[provider_name] = provider
                        else: raise NotImplementedError
                return self._objects[provider_name]

        @log_with()
        def build_best_prediction(self):
                print("Building advanced multiclass RF model with outliers removed!")
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.multiclass import OneVsRestClassifier
                from sklearn.multiclass import OneVsOneClassifier
                
                data_provider = self.get_data_provider(self._configuration['model']['data_provider'])
                target_variable_names = self._configuration['model']['target']
                input_features_names = self._configuration['model']['input_features']

                self.my_model = RandomForestClassifier(criterion=self._configuration['model']['criterion'],
                                                       class_weight=self._configuration['model']['class_weight'])
                # use a full grid over all parameters
                param_grid = {'estimator__n_estimators': self._configuration['model']['n_estimators'],
                              'estimator__min_samples_leaf':self._configuration['model']['min_samples_leaf'],
                              'estimator__max_depth':self._configuration['model']['max_depth']}

                # run grid search
                from sklearn.model_selection import GridSearchCV
                grid_search = GridSearchCV(OneVsRestClassifier(self.my_model), param_grid=param_grid,return_train_score=True,scoring='roc_auc_ovo')
                start = time.time()
                print(data_provider.train[target_variable_names])
                grid_search.fit(data_provider.train[input_features_names], data_provider.train[target_variable_names])
                print("GridSearchCV took %.2f seconds for %d candidate parameter settings." % (time.time() - start, len(grid_search.cv_results_['params'])))

                report(grid_search.cv_results_)

                self.my_model = grid_search.best_estimator_
                self.fit_results = grid_search.cv_results_

                # self.fit_results = self.my_model.evals_result()
                pickle.dump(self.my_model, open(self._configuration['model']['output_filename'], 'wb'))

                pass

class AdvancedModelClassificationRF(Model):
        def __init__(self,configuration):
                self._configuration = configuration
                self._objects = {}
                self._annotation = 'Performance comparision of different MVA discriminants'
                if 'annotation' in self._configuration:
                        self._annotation = self._configuration['annotation']
                self.do_training = self._configuration['model'].get('do_training',False)
                self.fit_results = None
                self.my_model = None
                self.Initialize()

        @log_with()
        def Initialize(self):
                if self.do_training: self.build_best_prediction()
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
        def get_data_provider(self,provider_name):
                """
                Factory method for data providers
                """
                if provider_name in self._objects:
                        return self._objects[provider_name]
                else:
                        if '.csv' in self._configuration[provider_name]['input_file']:
                                if self._configuration[provider_name]['type'] =='PandasDataProviderRespondingClients': 
                                        raise NotImplementedError
                                elif self._configuration[provider_name]['type'] =='PandasDataProviderRespondingClientsNoOutliers': 
                                        provider = PandasDataProviderRespondingClientsNoOutliers(self._configuration[provider_name]['input_file'],
                                        remove_all=self._configuration[provider_name]['remove_all'],training_set=self._configuration[provider_name]['training_set'])
                                else: raise NotImplementedError
                                self._objects[provider_name] = provider
                        else: raise NotImplementedError
                return self._objects[provider_name]

        @log_with()
        def build_best_prediction(self):
                print("Building advanced multiclass RF model with outliers removed!")
                from sklearn.ensemble import RandomForestClassifier
                
                data_provider = self.get_data_provider(self._configuration['model']['data_provider'])

                target_variable_names = self._configuration['model']['target']
                input_features_names = self._configuration['model']['input_features']

                self.my_model = RandomForestClassifier(criterion=self._configuration['model']['criterion'],
                                                       class_weight=self._configuration['model']['class_weight'])
                # use a full grid over all parameters
                param_grid = {'n_estimators': self._configuration['model']['n_estimators'],
                              'min_samples_leaf':self._configuration['model']['min_samples_leaf'],
                              'max_depth':self._configuration['model']['max_depth']}

                # run grid search
                from sklearn.model_selection import GridSearchCV
                grid_search = GridSearchCV(self.my_model, param_grid=param_grid,return_train_score=True,scoring='roc_auc_ovr')
                start = time.time()
                grid_search.fit(data_provider.train[input_features_names], np.ravel(data_provider.train[target_variable_names]))
                print("GridSearchCV took %.2f seconds for %d candidate parameter settings." % (time.time() - start, len(grid_search.cv_results_['params'])))

                report(grid_search.cv_results_)

                self.my_model = grid_search.best_estimator_

                # self.fit_results = self.my_model.evals_result()
                pickle.dump(self.my_model, open(self._configuration['model']['output_filename'], 'wb'))

                pass

class AdvancedModelClassificationMLP(Model):
        def __init__(self,configuration):
                self._configuration = configuration
                self._objects = {}
                self._annotation = 'Performance comparision of different MVA discriminants'
                if 'annotation' in self._configuration:
                        self._annotation = self._configuration['annotation']
                self.do_training = self._configuration['model'].get('do_training',False)
                self.fit_results = None
                self.my_model = None
                self.Initialize()

        @log_with()
        def Initialize(self):
                if self.do_training: self.build_best_prediction()
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
        def get_data_provider(self,provider_name):
                """
                Factory method for data providers
                """
                if provider_name in self._objects:
                        return self._objects[provider_name]
                else:
                        if '.csv' in self._configuration[provider_name]['input_file']:
                                if self._configuration[provider_name]['type'] =='PandasDataProviderRespondingClients': 
                                        raise NotImplementedError
                                elif self._configuration[provider_name]['type'] =='PandasDataProviderRespondingClientsNoOutliers': 
                                        provider = PandasDataProviderRespondingClientsNoOutliers(self._configuration[provider_name]['input_file'],
                                        remove_all=self._configuration[provider_name]['remove_all'],training_set=self._configuration[provider_name]['training_set'])
                                else: raise NotImplementedError
                                self._objects[provider_name] = provider
                        else: raise NotImplementedError
                return self._objects[provider_name]
  
        @log_with()
        def build_best_prediction(self):
                print("Building advanced multiclass MLP model with outliers removed!")
                from sklearn.neural_network import MLPClassifier
                from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
                
                target_variable_names = self._configuration['model']['target']
                data_provider = self.get_data_provider(self._configuration['model']['data_provider'])

                input_features_names = self._configuration['model']['input_features']
                X_train = data_provider.train[input_features_names]
                y_train = data_provider.train[target_variable_names]

                X_test = data_provider.test[input_features_names]
                y_test = data_provider.test[target_variable_names]

                self.my_model = MLPClassifier(activation=self._configuration['model']['activation'])

                # use a full grid over all parameters
                param_grid = {'estimator__alpha': self._configuration['model']['alpha'],
                              'estimator__max_iter':self._configuration['model']['max_iter'],
                              'estimator__hidden_layer_sizes':self._configuration['model']['hidden_layer_sizes']}

                # run grid search
                from sklearn.model_selection import GridSearchCV
                from sklearn.multiclass import OneVsRestClassifier
                grid_search = GridSearchCV(OneVsRestClassifier(self.my_model), param_grid=param_grid,return_train_score=True,scoring='roc_auc_ovr')
                start = time.time()
                grid_search.fit(data_provider.train[input_features_names],  data_provider.train[target_variable_names])
                print("GridSearchCV took %.2f seconds for %d candidate parameter settings." % (time.time() - start, len(grid_search.cv_results_['params'])))

                report(grid_search.cv_results_)

                self.my_model = grid_search.best_estimator_

                self.my_model.fit(X_train, y_train)

                # y_pred = self.my_model.predict(X_test)
                # print "Max error: ", max_error(y_test,y_pred)
                # print "Explained variance score: ", explained_variance_score(y_test,y_pred)
                # print "Mean absolute error: ", mean_absolute_error(y_test,y_pred)
                # print "Mean squared error: ", mean_squared_error(y_test,y_pred)

                # self.fit_results = self.my_model.evals_result()
                self.fit_results = grid_search.cv_results_

                pickle.dump(self.my_model, open(self._configuration['model']['output_filename'], 'wb'))

                pass

class AdvancedModelClassificationSVC(Model):
        def __init__(self,configuration):
                self._configuration = configuration
                self._objects = {}
                self._annotation = 'Performance comparision of different MVA discriminants'
                if 'annotation' in self._configuration:
                        self._annotation = self._configuration['annotation']
                self.do_training = self._configuration['model'].get('do_training',False)
                self.fit_results = None
                self.my_model = None
                self.Initialize()

        @log_with()
        def Initialize(self):
                if self.do_training: self.build_best_prediction()
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
        def get_data_provider(self,provider_name):
                """
                Factory method for data providers
                """
                if provider_name in self._objects:
                        return self._objects[provider_name]
                else:
                        if '.csv' in self._configuration[provider_name]['input_file']:
                                if self._configuration[provider_name]['type'] =='PandasDataProviderRespondingClients': 
                                        raise NotImplementedError
                                elif self._configuration[provider_name]['type'] =='PandasDataProviderRespondingClientsNoOutliers': 
                                        provider = PandasDataProviderRespondingClientsNoOutliers(self._configuration[provider_name]['input_file'],
                                        remove_all=self._configuration[provider_name]['remove_all'],training_set=self._configuration[provider_name]['training_set'])
                                else: raise NotImplementedError
                                self._objects[provider_name] = provider
                        else: raise NotImplementedError
                return self._objects[provider_name]
  
        @log_with()
        def build_best_prediction(self):
                print("Building advanced multiclass SVC model with outliers removed!")
                from sklearn import svm
                from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
                
                target_variable_names = self._configuration['model']['target']
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

                # eval_set = [(X_train, y_train), (X_test, y_test)]

                self.my_model = svm.SVC(kernel=self._configuration['model']['kernel'], C=self._configuration['model']['C'])
                self.my_model.fit(X_train, y_train)

                # y_pred = self.my_model.predict(X_test)
                # print "Max error: ", max_error(y_test,y_pred)
                # print "Explained variance score: ", explained_variance_score(y_test,y_pred)
                # print "Mean absolute error: ", mean_absolute_error(y_test,y_pred)
                # print "Mean squared error: ", mean_squared_error(y_test,y_pred)

                # self.fit_results = self.my_model.evals_result()
                pickle.dump(self.my_model, open(self._configuration['model']['output_filename'], 'wb'))

                pass

class PredictionModel(Model):
        def __init__(self,configuration):
                self._configuration = configuration
                self._objects = {}
                self._annotation = 'Predictions for KBC test task'
                if 'annotation' in self._configuration:
                        self._annotation = self._configuration['annotation']
                self.do_training = self._configuration['model'].get('do_training',False)
                self.my_multiclassification = None
                self.my_revenue_mf = None
                self.my_revenue_cc = None
                self.my_revenue_cl = None

                self.Initialize()

        @log_with()
        def Initialize(self):
                self.my_multiclassification = pickle.load(open(self._configuration['model']['multiclass_model'], 'rb'))
                self.my_revenue_mf = pickle.load(open(self._configuration['model']['regression_MF'], 'rb'))
                self.my_revenue_cc = pickle.load(open(self._configuration['model']['regression_CC'], 'rb'))
                self.my_revenue_cl = pickle.load(open(self._configuration['model']['regression_CL'], 'rb'))
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
        def get_data_provider(self,provider_name):
                """
                Factory method for data providers
                """
                if provider_name in self._objects:
                        return self._objects[provider_name]
                else:
                        if '.csv' in self._configuration[provider_name]['input_file']:
                                if self._configuration[provider_name]['type'] =='PandasDataProviderRespondingClients': 
                                        raise NotImplementedError
                                elif self._configuration[provider_name]['type'] =='PandasDataProviderRespondingClientsNoOutliers': 
                                        provider = PandasDataProviderRespondingClientsNoOutliers(self._configuration[provider_name]['input_file'],
                                        remove_all=self._configuration[provider_name]['remove_all'],training_set=self._configuration[provider_name]['training_set'])
                                else: raise NotImplementedError
                                self._objects[provider_name] = provider
                        else: raise NotImplementedError
                return self._objects[provider_name]

class AdvancedModelClassification(Model):
        def __init__(self,configuration):
                self._configuration = configuration
                self._objects = {}
                self._annotation = 'Performance comparision of different MVA discriminants'
                if 'annotation' in self._configuration:
                        self._annotation = self._configuration['annotation']
                self.do_training = self._configuration['model'].get('do_training',False)
                self.fit_results = None
                self.my_model = None
                self.Initialize()

        @log_with()
        def Initialize(self):
                if self.do_training: self.build_best_prediction()
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
        def get_data_provider(self,provider_name):
                """
                Factory method for data providers
                """
                if provider_name in self._objects:
                        return self._objects[provider_name]
                else:
                        if '.csv' in self._configuration[provider_name]['input_file']:
                                if self._configuration[provider_name]['type'] =='PandasDataProviderRespondingClients': 
                                        raise NotImplementedError
                                elif self._configuration[provider_name]['type'] =='PandasDataProviderRespondingClientsNoOutliers': 
                                        provider = PandasDataProviderRespondingClientsNoOutliers(self._configuration[provider_name]['input_file'],
                                        remove_all=self._configuration[provider_name]['remove_all'],training_set=self._configuration[provider_name]['training_set'])
                                else: raise NotImplementedError
                                self._objects[provider_name] = provider
                        else: raise NotImplementedError
                return self._objects[provider_name]

        @log_with()
        def build_best_prediction(self):
                print("Building advanced multiclass XGBoost model with outliers removed!")
                from xgboost import XGBClassifier
                from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
                
                target_variable_names = self._configuration['model']['target']
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

                self.my_model = XGBClassifier(n_estimators=self._configuration['model']['n_estimators'],
                                        max_depth=self._configuration['model']['max_depth'],
                                        learning_rate=self._configuration['model']['learning_rate'],
                                        objective=self._configuration['model']['objective'],
                                        verbosity=0)
                self.my_model.fit(X_train, y_train, eval_metric=["merror", "mlogloss"], eval_set=eval_set, verbose=False)

                # y_pred = self.my_model.predict(X_test)
                # print "Max error: ", max_error(y_test,y_pred)
                # print "Explained variance score: ", explained_variance_score(y_test,y_pred)
                # print "Mean absolute error: ", mean_absolute_error(y_test,y_pred)
                # print "Mean squared error: ", mean_squared_error(y_test,y_pred)

                self.fit_results = self.my_model.evals_result()
                pickle.dump(self.my_model, open(self._configuration['model']['output_filename'], 'wb'))

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
                self.my_model = None
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
        def get_data_provider(self,provider_name):
                """
                Factory method for data providers
                """
                if provider_name in self._objects:
                        return self._objects[provider_name]
                else:
                        if '.csv' in self._configuration[provider_name]['input_file']:
                                if self._configuration[provider_name]['type'] =='PandasDataProviderRespondingClientsRevenueMF': 
                                        provider = PandasDataProviderRespondingClientsRevenueMF(self._configuration[provider_name]['input_file'],training_set=self._configuration[provider_name]['training_set'])
                                elif self._configuration[provider_name]['type'] =='PandasDataProviderRespondingClientsNoOutliersRevenueMF': 
                                        provider = PandasDataProviderRespondingClientsNoOutliersRevenueMF(self._configuration[provider_name]['input_file'],
                                        self._configuration[provider_name]['remove_all'],training_set=self._configuration[provider_name]['training_set'])
                                elif self._configuration[provider_name]['type'] =='PandasDataProviderRespondingClientsNoOutliersRevenueCC': 
                                        provider = PandasDataProviderRespondingClientsNoOutliersRevenueCC(self._configuration[provider_name]['input_file'],
                                        self._configuration[provider_name]['remove_all'],training_set=self._configuration[provider_name]['training_set'])
                                elif self._configuration[provider_name]['type'] =='PandasDataProviderRespondingClientsNoOutliersRevenueCL': 
                                        provider = PandasDataProviderRespondingClientsNoOutliersRevenueCL(self._configuration[provider_name]['input_file'],
                                        self._configuration[provider_name]['remove_all'],training_set=self._configuration[provider_name]['training_set'])
                                else: raise NotImplementedError
                                self._objects[provider_name] = provider
                        else: raise NotImplementedError
                return self._objects[provider_name]

        @log_with()
        def build_best_prediction(self):
                print("Dummy building vanilla model!")

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

                self.my_model = XGBRegressor(n_estimators=self._configuration['model']['n_estimators'],
                                        max_depth=self._configuration['model']['max_depth'],
                                        learning_rate=self._configuration['model']['learning_rate'],
                                        verbosity=0)
                self.my_model.fit(X_train, y_train, eval_metric=["rmse", "mae"], eval_set=eval_set, verbose=False)

                y_pred = self.my_model.predict(X_test)
                # print "Max error: ", max_error(y_test,y_pred)
                print ("Explained variance score: ", explained_variance_score(y_test,y_pred))
                print ("Mean absolute error: ", mean_absolute_error(y_test,y_pred))
                print ("Mean squared error: ", mean_squared_error(y_test,y_pred))

                self.fit_results = self.my_model.evals_result()
                # print 'YO importance'
                # plot_importance(my_model)
                pickle.dump(self.my_model, open(self._configuration['model']['output_filename'], 'wb'))

                pass

class VanillaModelRegression(Model):
        def __init__(self,configuration):
                self._configuration = configuration
                self._objects = {}
                self._annotation = 'Performance comparision of different MVA discriminants'
                if 'annotation' in self._configuration:
                        self._annotation = self._configuration['annotation']
                self.my_model = None
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
        def get_data_provider(self,provider_name):
                """
                Factory method for data providers
                """
                from dataprovider import PandasDataProviderFromCSV_original
                if provider_name in self._objects:
                        return self._objects[provider_name]
                else:
                        if '.csv' in self._configuration[provider_name]['input_file']:
                                provider = PandasDataProviderFromCSV_original(self._configuration[provider_name]['input_file'])
                                self._objects[provider_name] = provider
                        else: raise NotImplementedError
                return self._objects[provider_name]

        @log_with()
        def build_best_prediction(self):
                print ("Dummy building vanilla model!")

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

                self.my_model = XGBRegressor(n_estimators=self._configuration['model']['n_estimators'],
                                        max_depth=self._configuration['model']['max_depth'],
                                        learning_rate=self._configuration['model']['learning_rate'],
                                        verbosity=0)
                self.my_model.fit(X_train, y_train, eval_metric=["rmse", "mae"], eval_set=eval_set, verbose=False)

                y_pred = my_model.predict(X_test)
                # print "Max error: ", max_error(y_test,y_pred)
                print ("Explained variance score: ", explained_variance_score(y_test,y_pred))
                print ("Mean absolute error: ", mean_absolute_error(y_test,y_pred))
                print ("Mean squared error: ", mean_squared_error(y_test,y_pred))

                self.fit_results = self.my_model.evals_result()
                # print 'YO importance'
                # plot_importance(my_model)
                pickle.dump(self.my_model, open(self._configuration['model']['output_filename'], 'wb'))

                pass

class VanillaModelLassoLarsIC(Model):
        def __init__(self,configuration):
                self._configuration = configuration
                self._objects = {}
                self._annotation = 'Ridge linear regression with built-in CV'
                if 'annotation' in self._configuration:
                        self._annotation = self._configuration['annotation']
                self.my_model = None
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
        def get_data_provider(self,provider_name):
                """
                Factory method for data providers
                """
                from dataprovider import PandasDataProviderFromCSV_original
                if provider_name in self._objects:
                        return self._objects[provider_name]
                else:
                        if '.csv' in self._configuration[provider_name]['input_file']:
                                provider = PandasDataProviderFromCSV_original(self._configuration[provider_name]['input_file'])
                                self._objects[provider_name] = provider
                        else: raise NotImplementedError
                return self._objects[provider_name]

        @log_with()
        def build_best_prediction(self):
                print ("Building LassoLarsIC linear regression vanilla model!")

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
                print ("AIC Explained variance score: ", explained_variance_score(y_test,y_pred_aic))
                print ("AIC Mean absolute error: ", mean_absolute_error(y_test,y_pred_aic))
                print ("AIC Mean squared error: ", mean_squared_error(y_test,y_pred_aic))

                my_model_bic = LassoLarsIC(criterion='bic')
                my_model_bic.fit(X_train, y_train)
                y_pred_bic = my_model_bic.predict(X_test)
                # print "Max error: ", max_error(y_test,y_pred)
                print ("BIC Explained variance score: ", explained_variance_score(y_test,y_pred_bic))
                print ("BIC Mean absolute error: ", mean_absolute_error(y_test,y_pred_bic))
                print ("BIC Mean squared error: ", mean_squared_error(y_test,y_pred_bic))

                self.fit_results = {'aic':my_model_aic, 'bic':my_model_bic}
                pickle.dump(self.my_model, open(self._configuration['model']['output_filename'], 'wb'))

                pass
class VanillaModelClassification(Model):
        def __init__(self,configuration):
                self._configuration = configuration
                self._objects = {}
                self._annotation = 'Performance comparision of different MVA discriminants'
                if 'annotation' in self._configuration:
                        self._annotation = self._configuration['annotation']
                self.do_training = self._configuration['model'].get('do_training',False)
                self.my_model = None
                self.fit_results = None
                self.Initialize()

        @log_with()
        def Initialize(self):
                if self.do_training: self.build_best_prediction()
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
        def get_data_provider(self,provider_name):
                """
                Factory method for data providers
                """
                from dataprovider import PandasDataProviderFromCSV_original
                if provider_name in self._objects:
                        return self._objects[provider_name]
                else:
                        if '.csv' in self._configuration[provider_name]['input_file']:
                                if self._configuration[provider_name]['type'] =='PandasDataProviderRespondingClients': 
                                        raise NotImplementedError
                                elif self._configuration[provider_name]['type'] =='PandasDataProviderRespondingClientsNoOutliers': 
                                        provider = PandasDataProviderRespondingClientsNoOutliers(self._configuration[provider_name]['input_file'],
                                        remove_all=self._configuration[provider_name]['remove_all'],training_set=self._configuration[provider_name]['training_set'])
                                elif self._configuration[provider_name]['type'] =='PandasDataProviderFromCSV_TrainPredictionDatasetsForInclusive':
                                        provider = PandasDataProviderFromCSV_TrainPredictionDatasetsForInclusive(self._configuration[provider_name]['input_file'])
                                else:
                                        provider = PandasDataProviderFromCSV_original(self._configuration[provider_name]['input_file'])
                                self._objects[provider_name] = provider
                        else: raise NotImplementedError
                return self._objects[provider_name]

        @log_with()
        def build_best_prediction(self):
                print ("Dummy building vanilla model!")

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

                self.my_model = XGBClassifier(n_estimators=self._configuration['model']['n_estimators'],
                                        max_depth=self._configuration['model']['max_depth'],
                                        learning_rate=self._configuration['model']['learning_rate'],
                                        verbosity=0)
                self.my_model.fit(X_train, y_train, eval_metric=["error", "auc", "map"], eval_set=eval_set, verbose=False)

                # y_pred = my_model.predict(X_test)
                # print "Max error: ", max_error(y_test,y_pred)
                # print "Explained variance score: ", explained_variance_score(y_test,y_pred)
                # print "Mean absolute error: ", mean_absolute_error(y_test,y_pred)
                # print "Mean squared error: ", mean_squared_error(y_test,y_pred)

                self.fit_results = my_model.evals_result()
                # print 'YO importance'
                # plot_importance(my_model)
                pickle.dump(self.my_model, open(self._configuration['model']['output_filename'], 'wb'))

                pass

