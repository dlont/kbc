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
                from dataprovider import PandasDataProviderFromCSV_titanic
                if name in self._objects:
                        return self._objects[name]
                else:
                        if '.csv' in self._configuration[name]['input_file']:
                                provider = PandasDataProviderFromCSV_titanic(self._configuration[name]['input_file'])
                                self._objects[name] = provider
                        else: raise NotImplementedError
                return self._objects[name]

        @log_with()
        def style_features_correlation_pad(self, ax, feature_tup):
                '''
                Apply style such as logarithmic scales, legend, title, etc.
                '''
                # Plot style
                if 'legend' in self._configuration[feature_tup[0]]['style']: 
                        ax.legend(prop=self._configuration[feature_tup[0]]['style']['legend']['prop'])
                # if 'logx' in self._configuration[feature_tup[0]]['style']: 
                #         if self._configuration[feature_tup[0]]['style']['logx']: ax.set_xscale("log")
                # if 'logy' in self._configuration[feature_tup[1]]['style']: 
                #         if self._configuration[feature_tup[1]]['style']['logy']: ax.set_yscale("log")
                ax.set_title('{0}:{1}'.format(*feature_tup))

        @log_with()
        def style_feature_pad(self, ax, feature_name):
                '''
                Apply style such as logarithmic scales, legend, title, etc.
                '''
                # Plot style
                if 'legend' in self._configuration[feature_name]['style']: 
                        ax.legend(prop=self._configuration[feature_name]['style']['legend']['prop'])
                if 'logx' in self._configuration[feature_name]['style']: 
                        if self._configuration[feature_name]['style']['logx']: ax.set_xscale("log")
                if 'logy' in self._configuration[feature_name]['style']: 
                        if self._configuration[feature_name]['style']['logy']: ax.set_yscale("log")
                ax.set_title(self._configuration[feature_name]['title'])

        @log_with()
        def build_train_correlation_pad(self, ax, feature_tup):
                '''
                Plot training scatter plot for two features
                ax: matplotlib axes instance
                feature_tup: two elements tuple with features names
                '''
                logging.debug('Plotting features: {0}'.format(feature_tup))

                from dataprovider import PandasSurvivedClassSelector, PandasDrownedClassSelector
                data_provider_f0 = self.get_data_provider(self._configuration[feature_tup[0]]['data_provider'])
                class1_selector_f0 = self._configuration[feature_tup[0]]['class1']
                class2_selector_f0  = self._configuration[feature_tup[0]]['class2']

                data_provider_f1 = self.get_data_provider(self._configuration[feature_tup[1]]['data_provider'])
                class1_selector_f1 = self._configuration[feature_tup[1]]['class1']
                class2_selector_f1  = self._configuration[feature_tup[1]]['class2']

                class1_training_f0 = list(data_provider_f0.get_training_examples(feature_tup[0],class1_selector_f0))
                class1_training_f1 = list(data_provider_f1.get_training_examples(feature_tup[1],class1_selector_f1))
                ax.scatter(class1_training_f0, class1_training_f1,
                                color=self._configuration[feature_tup[0]]['class1_color_train'],
                                label=self._configuration[feature_tup[0]]['class1_label_train'], alpha = 0.5)

                class2_training_f0 = list(data_provider_f0.get_training_examples(feature_tup[0],class2_selector_f0))
                class2_training_f1 = list(data_provider_f1.get_training_examples(feature_tup[1],class2_selector_f1))
                ax.scatter(class2_training_f0, class2_training_f1,
                                color=self._configuration[feature_tup[0]]['class2_color_train'],
                                label=self._configuration[feature_tup[0]]['class2_label_train'], alpha = 0.5)

                # Plot style
                self.style_features_correlation_pad(ax,feature_tup)

        @log_with()      
        def build_test_train_categorical_feature_pad(self, ax, feature_name):
                '''
                Plot training and testing distributions of a feature
                ax: matplotlib axes instance
                feature_name: name of the feature distribution to plot
                '''
                logging.debug('Plotting feature: {0}'.format(feature_name))

                from dataprovider import PandasSurvivedClassSelector, PandasDrownedClassSelector
                data_provider = self.get_data_provider(self._configuration[feature_name]['data_provider'])
                class1_selector = self._configuration[feature_name]['class1']
                class2_selector  = self._configuration[feature_name]['class2']
        
                class1_training = list(data_provider.get_training_examples(feature_name,class1_selector).fillna('Unkw'))
                class1_testing = list(data_provider.get_testing_examples(feature_name,class1_selector).fillna('Unkw'))
                class2_training = list(data_provider.get_training_examples(feature_name,class2_selector).fillna('Unkw'))
                class2_testing = list(data_provider.get_testing_examples(feature_name,class2_selector).fillna('Unkw'))
                
                # Binning configuration
                bins = self._configuration[feature_name]['style']['bins']

                class1_training_dic = {bin:0 for bin in bins}
                for entry in class1_training: class1_training_dic[entry]+=1
                for bin in bins: class1_training_dic[bin]/=float(len(class1_training))
                # sort entries in the dictionary in a way defined by the config file
                class1_training_values = [class1_training_dic[bin] for bin in bins]
                
                class2_training_dic = {bin:0 for bin in bins}
                for entry in class2_training: class2_training_dic[entry]+=1
                for bin in bins: class2_training_dic[bin]/=float(len(class2_training))
                # sort entries in the dictionary in a way defined by the config file
                class2_training_values = [class2_training_dic[bin] for bin in bins]

                # Class 1 training and testing distributions
                ax.bar(bins, class1_training_values, color=self._configuration[feature_name]['class1_color_train'], 
                        label=self._configuration[feature_name]['class1_label_train'], alpha = 0.5)
                class1_testing_dic = {bin:0 for bin in bins}
                class1_testing_err_dic = {bin:0 for bin in bins}
                for entry in class1_testing: class1_testing_dic[entry]+=1
                for bin in bins: class1_testing_err_dic[bin]=np.sqrt(class1_testing_dic[bin])
                for bin in bins: 
                        class1_testing_dic[bin]/=float(len(class1_testing))
                        class1_testing_err_dic[bin]/=float(len(class1_testing))
                # sort entries in the dictionary in a way defined by the config file
                class1_testing_values = [class1_testing_dic[bin] for bin in bins]
                class1_testing_values_err = [class1_testing_err_dic[bin] for bin in bins]
                ax.errorbar(bins, class1_testing_values, yerr=class1_testing_values_err,
                                marker=self._configuration[feature_name]['class1_marker_test'], 
                                ls=self._configuration[feature_name]['class1_line_test'],
                                color=self._configuration[feature_name]['class1_color_test'], 
                                label=self._configuration[feature_name]['class1_label_test'])

                # Class 2 training and testing distributions
                ax.bar(bins, class2_training_values, color=self._configuration[feature_name]['class2_color_train'], 
                label=self._configuration[feature_name]['class2_label_train'], alpha = 0.5)
                class2_testing_dic = {bin:0 for bin in bins}
                class2_testing_err_dic = {bin:0 for bin in bins}
                for entry in class2_testing: class2_testing_dic[entry]+=1
                for bin in bins: class2_testing_err_dic[bin]=np.sqrt(class2_testing_dic[bin])
                for bin in bins: 
                        class2_testing_dic[bin]/=float(len(class2_testing))
                        class2_testing_err_dic[bin]/=float(len(class2_testing))
                # sort entries in the dictionary in a way defined by the config file
                class2_testing_values = [class2_testing_dic[bin] for bin in bins]
                class2_testing_values_err = [class2_testing_err_dic[bin] for bin in bins]
                ax.errorbar(bins, class2_testing_values, yerr=class2_testing_values_err,
                                marker=self._configuration[feature_name]['class2_marker_test'], 
                                ls=self._configuration[feature_name]['class2_line_test'],
                                color=self._configuration[feature_name]['class2_color_test'], 
                                label=self._configuration[feature_name]['class2_label_test'])
                
                # Plot style
                self.style_feature_pad(ax,feature_name)

        @log_with()      
        def build_test_train_numerical_feature_pad(self, ax, feature_name):
                '''
                Plot training and testing distributions of a feature
                ax: matplotlib axes instance
                feature_name: name of the feature distribution to plot
                '''
                logging.debug('Plotting feature: {0}'.format(feature_name))

                from dataprovider import PandasSurvivedClassSelector, PandasDrownedClassSelector
                data_provider = self.get_data_provider(self._configuration[feature_name]['data_provider'])
                class1_selector = self._configuration[feature_name]['class1']
                class2_selector  = self._configuration[feature_name]['class2']
        
                class1_training = list(data_provider.get_training_examples(feature_name,class1_selector).fillna(-1))
                class1_testing = list(data_provider.get_testing_examples(feature_name,class1_selector).fillna(-1))
                class2_training = list(data_provider.get_training_examples(feature_name,class2_selector).fillna(-1))
                class2_testing = list(data_provider.get_testing_examples(feature_name,class2_selector).fillna(-1))
                
                # Binning configuration
                underflow, overflow = self._configuration[feature_name]['style']['under_over_flow']
                bins = self._configuration[feature_name]['style']['bins']
                bin_centers = bins[0:-1]+np.diff(bins)/2.
                logging.debug('bin_centers ({0})={{1}}'.format(feature_name,bin_centers))

                # Class 1 training and testing distributions
                if any([underflow, overflow]):
                        ax.hist(np.clip(class1_training, bins[0] if underflow else None, bins[-1] if overflow else None), bins,
                        density=True, histtype='stepfilled',
                        color=self._configuration[feature_name]['class1_color_train'], 
                        label=self._configuration[feature_name]['class1_label_train'], alpha = 0.5)
                else:
                        ax.hist(class1_training, bins, 
                        density=True, histtype='stepfilled', 
                        color=self._configuration[feature_name]['class1_color_train'], 
                        label=self._configuration[feature_name]['class1_label_train'], alpha = 0.5)
                hist_testing = np.histogram(class1_testing, bins)
                if any([underflow, overflow]):
                        hist_testing = np.histogram(np.clip(class1_testing, bins[0] if underflow else None, bins[-1] if overflow else None), bins)
                points_testing_y = hist_testing[0]/np.diff(bins)/float(np.sum(hist_testing[0]))
                points_testing_yerr = np.sqrt(hist_testing[0])/np.diff(bins)/float(np.sum(hist_testing[0]))
                ax.errorbar(bin_centers, points_testing_y, yerr=points_testing_yerr, marker=self._configuration[feature_name]['class1_marker_test'], 
                                ls=self._configuration[feature_name]['class1_line_test'], color=self._configuration[feature_name]['class1_color_test'], 
                                label=self._configuration[feature_name]['class1_label_test'])

                # Class 2 training and testing distributions
                if any([underflow, overflow]):
                        ax.hist(np.clip(class2_training, bins[0] if underflow else None, bins[-1] if overflow else None), bins,
                        density=True, histtype='stepfilled',
                        color=self._configuration[feature_name]['class2_color_train'], 
                        label=self._configuration[feature_name]['class2_label_train'], alpha = 0.5)
                else:
                        ax.hist(class2_training, bins,
                        density=True, histtype='stepfilled',
                        color=self._configuration[feature_name]['class2_color_train'], 
                        label=self._configuration[feature_name]['class2_label_train'], alpha = 0.5)
                        
                hist_testing = np.histogram(class2_testing, bins)
                if any([underflow, overflow]):
                        hist_testing = np.histogram(np.clip(class2_testing, bins[0] if underflow else None, bins[-1] if overflow else None), bins)
                points_testing_y = hist_testing[0]/np.diff(bins)/float(np.sum(hist_testing[0]))
                points_testing_yerr = np.sqrt(hist_testing[0])/np.diff(bins)/np.sum(hist_testing[0])
                ax.errorbar(bin_centers, points_testing_y, yerr=points_testing_yerr, marker=self._configuration[feature_name]['class2_marker_test'], 
                                ls=self._configuration[feature_name]['class2_line_test'], color=self._configuration[feature_name]['class2_color_test'], 
                                label=self._configuration[feature_name]['class2_label_test'])
                
                # Plot style
                self.style_feature_pad(ax,feature_name)

        @log_with()
        def build_best_prediction(self):
                from xgboost import XGBClassifier
                from sklearn.metrics import classification_report
                from dataprovider import PandasSurvivedClassSelector, PandasDrownedClassSelector
                
                data_provider = self.get_data_provider(self._configuration['Age']['data_provider'])

                my_model = XGBClassifier()
                X_train, y_train = data_provider.train.drop(['Cabin','Ticket','Name','Survived'],axis=1),data_provider.train['Survived']
                # print X_train.dtypes

                my_model.fit(X_train, y_train)

                X_test, y_test = data_provider.test.drop(['Cabin','Ticket','Name','Survived'],axis=1),data_provider.test['Survived']
                y_pred = my_model.predict(X_test)
                print classification_report(y_test, y_pred, target_names=['Survived','Drowned'])

                self.predict_kaggle_output(my_model)

                pass

        @log_with()
        def predict_kaggle_output(self,ml_model):
                from sklearn.preprocessing import OneHotEncoder
                kaggle_input = pd.read_csv('data/2019-12-05/test.csv', index_col='PassengerId')
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
                from dataprovider import PandasDataProviderFromCSV_titanic_original
                if name in self._objects:
                        return self._objects[name]
                else:
                        if '.csv' in self._configuration[name]['input_file']:
                                provider = PandasDataProviderFromCSV_titanic_original(self._configuration[name]['input_file'])
                                self._objects[name] = provider
                        else: raise NotImplementedError
                return self._objects[name]

        @log_with()
        def style_features_correlation_pad(self, ax, feature_tup):
                '''
                Apply style such as logarithmic scales, legend, title, etc.
                '''
                # Plot style
                if 'legend' in self._configuration[feature_tup[0]]['style']: 
                        ax.legend(prop=self._configuration[feature_tup[0]]['style']['legend']['prop'])
                # if 'logx' in self._configuration[feature_tup[0]]['style']: 
                #         if self._configuration[feature_tup[0]]['style']['logx']: ax.set_xscale("log")
                # if 'logy' in self._configuration[feature_tup[1]]['style']: 
                #         if self._configuration[feature_tup[1]]['style']['logy']: ax.set_yscale("log")
                ax.set_title('{0}:{1}'.format(*feature_tup))

        @log_with()
        def style_feature_pad(self, ax, feature_name):
                '''
                Apply style such as logarithmic scales, legend, title, etc.
                '''
                # Plot style
                if 'legend' in self._configuration[feature_name]['style']: 
                        ax.legend(prop=self._configuration[feature_name]['style']['legend']['prop'])
                if 'logx' in self._configuration[feature_name]['style']: 
                        if self._configuration[feature_name]['style']['logx']: ax.set_xscale("log")
                if 'logy' in self._configuration[feature_name]['style']: 
                        if self._configuration[feature_name]['style']['logy']: ax.set_yscale("log")
                ax.set_title(self._configuration[feature_name]['title'])

        @log_with()
        def build_train_correlation_pad(self, ax, feature_tup):
                '''
                Plot training scatter plot for two features
                ax: matplotlib axes instance
                feature_tup: two elements tuple with features names
                '''
                logging.debug('Plotting features: {0}'.format(feature_tup))

                from dataprovider import PandasSurvivedClassSelector, PandasDrownedClassSelector
                data_provider_f0 = self.get_data_provider(self._configuration[feature_tup[0]]['data_provider'])
                class1_selector_f0 = self._configuration[feature_tup[0]]['class1']
                class2_selector_f0  = self._configuration[feature_tup[0]]['class2']

                data_provider_f1 = self.get_data_provider(self._configuration[feature_tup[1]]['data_provider'])
                class1_selector_f1 = self._configuration[feature_tup[1]]['class1']
                class2_selector_f1  = self._configuration[feature_tup[1]]['class2']

                class1_training_f0 = list(data_provider_f0.get_training_examples(feature_tup[0],class1_selector_f0))
                class1_training_f1 = list(data_provider_f1.get_training_examples(feature_tup[1],class1_selector_f1))
                ax.scatter(class1_training_f0, class1_training_f1,
                                color=self._configuration[feature_tup[0]]['class1_color_train'],
                                label=self._configuration[feature_tup[0]]['class1_label_train'], alpha = 0.5)

                class2_training_f0 = list(data_provider_f0.get_training_examples(feature_tup[0],class2_selector_f0))
                class2_training_f1 = list(data_provider_f1.get_training_examples(feature_tup[1],class2_selector_f1))
                ax.scatter(class2_training_f0, class2_training_f1,
                                color=self._configuration[feature_tup[0]]['class2_color_train'],
                                label=self._configuration[feature_tup[0]]['class2_label_train'], alpha = 0.5)

                # Plot style
                self.style_features_correlation_pad(ax,feature_tup)

        @log_with()      
        def build_test_train_categorical_feature_pad(self, ax, feature_name):
                '''
                Plot training and testing distributions of a feature
                ax: matplotlib axes instance
                feature_name: name of the feature distribution to plot
                '''
                logging.debug('Plotting feature: {0}'.format(feature_name))

                from dataprovider import PandasSurvivedClassSelector, PandasDrownedClassSelector
                data_provider = self.get_data_provider(self._configuration[feature_name]['data_provider'])
                class1_selector = self._configuration[feature_name]['class1']
                class2_selector  = self._configuration[feature_name]['class2']
        
                class1_training = list(data_provider.get_training_examples(feature_name,class1_selector).fillna('Unkw'))
                class1_testing = list(data_provider.get_testing_examples(feature_name,class1_selector).fillna('Unkw'))
                class2_training = list(data_provider.get_training_examples(feature_name,class2_selector).fillna('Unkw'))
                class2_testing = list(data_provider.get_testing_examples(feature_name,class2_selector).fillna('Unkw'))
                
                # Binning configuration
                bins = self._configuration[feature_name]['style']['bins']

                class1_training_dic = {bin:0 for bin in bins}
                for entry in class1_training: class1_training_dic[entry]+=1
                for bin in bins: class1_training_dic[bin]/=float(len(class1_training))
                # sort entries in the dictionary in a way defined by the config file
                class1_training_values = [class1_training_dic[bin] for bin in bins]
                
                class2_training_dic = {bin:0 for bin in bins}
                for entry in class2_training: class2_training_dic[entry]+=1
                for bin in bins: class2_training_dic[bin]/=float(len(class2_training))
                # sort entries in the dictionary in a way defined by the config file
                class2_training_values = [class2_training_dic[bin] for bin in bins]

                # Class 1 training and testing distributions
                ax.bar(bins, class1_training_values, color=self._configuration[feature_name]['class1_color_train'], 
                        label=self._configuration[feature_name]['class1_label_train'], alpha = 0.5)
                class1_testing_dic = {bin:0 for bin in bins}
                class1_testing_err_dic = {bin:0 for bin in bins}
                for entry in class1_testing: class1_testing_dic[entry]+=1
                for bin in bins: class1_testing_err_dic[bin]=np.sqrt(class1_testing_dic[bin])
                for bin in bins: 
                        class1_testing_dic[bin]/=float(len(class1_testing))
                        class1_testing_err_dic[bin]/=float(len(class1_testing))
                # sort entries in the dictionary in a way defined by the config file
                class1_testing_values = [class1_testing_dic[bin] for bin in bins]
                class1_testing_values_err = [class1_testing_err_dic[bin] for bin in bins]
                ax.errorbar(bins, class1_testing_values, yerr=class1_testing_values_err,
                                marker=self._configuration[feature_name]['class1_marker_test'], 
                                ls=self._configuration[feature_name]['class1_line_test'],
                                color=self._configuration[feature_name]['class1_color_test'], 
                                label=self._configuration[feature_name]['class1_label_test'])

                # Class 2 training and testing distributions
                ax.bar(bins, class2_training_values, color=self._configuration[feature_name]['class2_color_train'], 
                label=self._configuration[feature_name]['class2_label_train'], alpha = 0.5)
                class2_testing_dic = {bin:0 for bin in bins}
                class2_testing_err_dic = {bin:0 for bin in bins}
                for entry in class2_testing: class2_testing_dic[entry]+=1
                for bin in bins: class2_testing_err_dic[bin]=np.sqrt(class2_testing_dic[bin])
                for bin in bins: 
                        class2_testing_dic[bin]/=float(len(class2_testing))
                        class2_testing_err_dic[bin]/=float(len(class2_testing))
                # sort entries in the dictionary in a way defined by the config file
                class2_testing_values = [class2_testing_dic[bin] for bin in bins]
                class2_testing_values_err = [class2_testing_err_dic[bin] for bin in bins]
                ax.errorbar(bins, class2_testing_values, yerr=class2_testing_values_err,
                                marker=self._configuration[feature_name]['class2_marker_test'], 
                                ls=self._configuration[feature_name]['class2_line_test'],
                                color=self._configuration[feature_name]['class2_color_test'], 
                                label=self._configuration[feature_name]['class2_label_test'])
                
                # Plot style
                self.style_feature_pad(ax,feature_name)

        @log_with()      
        def build_test_train_numerical_feature_pad(self, ax, feature_name):
                '''
                Plot training and testing distributions of a feature
                ax: matplotlib axes instance
                feature_name: name of the feature distribution to plot
                '''
                logging.debug('Plotting feature: {0}'.format(feature_name))

                from dataprovider import PandasSurvivedClassSelector, PandasDrownedClassSelector
                data_provider = self.get_data_provider(self._configuration[feature_name]['data_provider'])
                class1_selector = self._configuration[feature_name]['class1']
                class2_selector  = self._configuration[feature_name]['class2']
        
                class1_training = list(data_provider.get_training_examples(feature_name,class1_selector).fillna(-1))
                class1_testing = list(data_provider.get_testing_examples(feature_name,class1_selector).fillna(-1))
                class2_training = list(data_provider.get_training_examples(feature_name,class2_selector).fillna(-1))
                class2_testing = list(data_provider.get_testing_examples(feature_name,class2_selector).fillna(-1))
                
                # Binning configuration
                underflow, overflow = self._configuration[feature_name]['style']['under_over_flow']
                bins = self._configuration[feature_name]['style']['bins']
                bin_centers = bins[0:-1]+np.diff(bins)/2.
                logging.debug('bin_centers ({0})={{1}}'.format(feature_name,bin_centers))

                # Class 1 training and testing distributions
                if any([underflow, overflow]):
                        ax.hist(np.clip(class1_training, bins[0] if underflow else None, bins[-1] if overflow else None), bins,
                        density=True, histtype='stepfilled',
                        color=self._configuration[feature_name]['class1_color_train'], 
                        label=self._configuration[feature_name]['class1_label_train'], alpha = 0.5)
                else:
                        ax.hist(class1_training, bins, 
                        density=True, histtype='stepfilled', 
                        color=self._configuration[feature_name]['class1_color_train'], 
                        label=self._configuration[feature_name]['class1_label_train'], alpha = 0.5)
                hist_testing = np.histogram(class1_testing, bins)
                if any([underflow, overflow]):
                        hist_testing = np.histogram(np.clip(class1_testing, bins[0] if underflow else None, bins[-1] if overflow else None), bins)
                points_testing_y = hist_testing[0]/np.diff(bins)/float(np.sum(hist_testing[0]))
                points_testing_yerr = np.sqrt(hist_testing[0])/np.diff(bins)/float(np.sum(hist_testing[0]))
                ax.errorbar(bin_centers, points_testing_y, yerr=points_testing_yerr, marker=self._configuration[feature_name]['class1_marker_test'], 
                                ls=self._configuration[feature_name]['class1_line_test'], color=self._configuration[feature_name]['class1_color_test'], 
                                label=self._configuration[feature_name]['class1_label_test'])

                # Class 2 training and testing distributions
                if any([underflow, overflow]):
                        ax.hist(np.clip(class2_training, bins[0] if underflow else None, bins[-1] if overflow else None), bins,
                        density=True, histtype='stepfilled',
                        color=self._configuration[feature_name]['class2_color_train'], 
                        label=self._configuration[feature_name]['class2_label_train'], alpha = 0.5)
                else:
                        ax.hist(class2_training, bins,
                        density=True, histtype='stepfilled',
                        color=self._configuration[feature_name]['class2_color_train'], 
                        label=self._configuration[feature_name]['class2_label_train'], alpha = 0.5)
                        
                hist_testing = np.histogram(class2_testing, bins)
                if any([underflow, overflow]):
                        hist_testing = np.histogram(np.clip(class2_testing, bins[0] if underflow else None, bins[-1] if overflow else None), bins)
                points_testing_y = hist_testing[0]/np.diff(bins)/float(np.sum(hist_testing[0]))
                points_testing_yerr = np.sqrt(hist_testing[0])/np.diff(bins)/np.sum(hist_testing[0])
                ax.errorbar(bin_centers, points_testing_y, yerr=points_testing_yerr, marker=self._configuration[feature_name]['class2_marker_test'], 
                                ls=self._configuration[feature_name]['class2_line_test'], color=self._configuration[feature_name]['class2_color_test'], 
                                label=self._configuration[feature_name]['class2_label_test'])
                
                # Plot style
                self.style_feature_pad(ax,feature_name)

        @log_with()
        def build_best_prediction(self):
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
                
                pass

