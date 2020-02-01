import os
import textwrap
import pprint as pp


import matplotlib.pyplot as plt

from logwith import *
from models import *

class View(object):
        @log_with()
        def __init__(self,view_name=None):
                self.view_name = view_name
                self.model = None
                self._style = None
                self._outfilename = 'out'
                self._outfileextension = 'png'
                self._outputfolder = 'build'
        @log_with()
        def set_style(self,style):
                self._style = style
        @log_with()
        def set_model(self,model):
                self.model = model
        @log_with()
        def set_builddir(self,folder):
                self._outputfolder = folder
                if not os.path.exists(folder):
                        os.makedirs(folder)
        @log_with()
        def set_outfilename(self,filename):
                if filename: self._outfilename = filename
        @log_with()
        def set_extension(self,extension):
                self._outfileextension = extension
        @log_with()
        def get_outfile_name(self,substring=''):
                for ext in self._outfileextension.split(","):
                        yield '{}/{}{}.{}'.format(self._outputfolder,self._outfilename,substring,ext)

        @log_with()
        def annotate(self,type):
                if type == "screen":
                        bright_green_text = "\033[1;32;40m"
                        normal_text = "\033[0;37;40m"
                        if self.view_name in self.model._configuration:
                                if 'annotation' in self.model._configuration[self.view_name]:
                                        print "\n".join(textwrap.wrap(bcolors.OKBLUE+
                                                self.model._configuration[self.view_name]['annotation'].encode('ascii')+
                                                bcolors.ENDC, 120))
                elif type == "tex":
                        logging.warning("Annotation format: {}. Not implemented yet!".format(type))
                elif type == "md":
                        logging.warning("Annotation format: {}. Not implemented yet!".format(type))
                else:
                        logging.error("Annotation format not recognized: {}".format(type))

        @log_with()
        def save_config(self, config, outputfolder=None):
                if outputfolder is None:
                        if os.path.exists(self._outputfolder):
                                # Writing configuration data
                                if "_cff.py" in config: 
                                        with open(self._outputfolder+'/'+os.path.basename(config), 'w') as f:
                                                serialized_config_str = pp.pformat(self.model._configuration)
                                                serialized_config_str = 'config='+serialized_config_str
                                                f.write(serialized_config_str)
                                elif ".json" in config: 
                                        with open(self._outputfolder+'/'+os.path.basename(config), 'w') as f:
                                                json.dump(self.model._configuration, f, indent=4, sort_keys=True)
                else:
                        if os.path.exists(outputfolder):
                                # Writing configuration data
                                if "_cff.py" in config: 
                                        with open(outputfolder+'/'+os.path.basename(config), 'w') as f:
                                                serialized_config_str = pp.pformat(self.model._configuration)
                                                serialized_config_str = 'config='+serialized_config_str
                                                f.write(serialized_config_str)
                                elif ".json" in config: 
                                        with open(outputfolder+'/'+os.path.basename(config), 'w') as f:
                                                json.dump(self.model._configuration, f, indent=4, sort_keys=True)

        @log_with()
        def save(self,serializer):
                serializer.serialize_view(self)

        @log_with()
        def draw(self):
                pass

class View2dCorrelationsTrain(View):
        @log_with()
        def __init__(self,view_name=None):
                self.view_name = view_name
                pass

        @log_with()
        def style_features_correlation_pad(self, ax, feature_tup):
                '''
                Apply style such as logarithmic scales, legend, title, etc.
                '''
                # Plot style
                if 'legend' in self.model._configuration[feature_tup[0]]['style']: 
                        ax.legend(prop=self.model._configuration[feature_tup[0]]['style']['legend']['prop'])
                # if 'logx' in self.model._configuration[feature_tup[0]]['style']: 
                #         if self.model._configuration[feature_tup[0]]['style']['logx']: ax.set_xscale("log")
                # if 'logy' in self.model._configuration[feature_tup[1]]['style']: 
                #         if self.model._configuration[feature_tup[1]]['style']['logy']: ax.set_yscale("log")
                ax.set_title('{0}:\n{1}'.format(*feature_tup))

        @log_with()
        def build_train_correlation_pad(self, ax, feature_tup):
                '''
                Plot training scatter plot for two features
                ax: matplotlib axes instance
                feature_tup: two elements tuple with features names
                '''
                logging.debug('Plotting features: {0}'.format(feature_tup))

                from dataprovider import PandasSurvivedClassSelector, PandasDrownedClassSelector
                data_provider_f0 = self.model.get_data_provider(self.model._configuration[feature_tup[0]]['data_provider'])
                class1_selector_f0 = self.model._configuration[feature_tup[0]]['class1']
                # class2_selector_f0  = self.model._configuration[feature_tup[0]]['class2']

                data_provider_f1 = self.model.get_data_provider(self.model._configuration[feature_tup[1]]['data_provider'])
                class1_selector_f1 = self.model._configuration[feature_tup[1]]['class1']
                # class2_selector_f1  = self.model._configuration[feature_tup[1]]['class2']

                class1_training_f0 = list(data_provider_f0.get_training_testing_examples(feature_tup[0],class1_selector_f0))
                class1_training_f1 = list(data_provider_f1.get_training_testing_examples(feature_tup[1],class1_selector_f1))
                ax.scatter(class1_training_f0, class1_training_f1,
                                color=self.model._configuration[feature_tup[0]]['class1_color_train'],
                                label=self.model._configuration[feature_tup[0]]['class1_label_train'], alpha = 0.5)

                # class2_training_f0 = list(data_provider_f0.get_training_testing_examples(feature_tup[0],class2_selector_f0))
                # class2_training_f1 = list(data_provider_f1.get_training_testing_examples(feature_tup[1],class2_selector_f1))
                # ax.scatter(class2_training_f0, class2_training_f1,
                #                 color=self.model._configuration[feature_tup[0]]['class2_color_train'],
                #                 label=self.model._configuration[feature_tup[0]]['class2_label_train'], alpha = 0.5)

                # Plot style
                self.style_features_correlation_pad(ax,feature_tup)

        @log_with()
        def draw(self):
                if not self.view_name: raise RuntimeError('Cannot build view. View name is not specified!')
                nrows=self.model._configuration[self.view_name]['layout']['nrows']
                ncols=self.model._configuration[self.view_name]['layout']['ncols']
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
                fig.set_size_inches(*self.model._configuration[self.view_name]['size'])
                pads = axes.flatten()

                for pad,features in enumerate(self.model._configuration[self.view_name]['features']):
                        if True:
                                self.build_train_correlation_pad(pads[pad],tuple(features))
                        else: 
                                # logging.error('Unknown feature type: {}'.format(self.model._configuration[feature]['style']['type']))
                                raise NotImplementedError

                fig.tight_layout()
                self.set_outfilename(self.model._configuration[self.view_name]['output_filename'])
                for name in self.get_outfile_name(): plt.savefig(name)
                plt.close(fig)

class View1dTrainTest(View):
        @log_with()
        def __init__(self,view_name=None):
                self.view_name = view_name
                pass

        @log_with()
        def style_feature_pad(self, ax, feature_name):
                '''
                Apply style such as logarithmic scales, legend, title, etc.
                '''
                # Plot style
                if 'legend' in self.model._configuration[feature_name]['style']: 
                        ax.legend(prop=self.model._configuration[feature_name]['style']['legend']['prop'])
                if 'logx' in self.model._configuration[feature_name]['style']: 
                        if self.model._configuration[feature_name]['style']['logx']: ax.set_xscale("log")
                if 'logy' in self.model._configuration[feature_name]['style']: 
                        if self.model._configuration[feature_name]['style']['logy']: ax.set_yscale("log")
                ax.set_title(self.model._configuration[feature_name]['title'])

        @log_with()      
        def build_test_train_numerical_zoom_feature_pad(self, ax, feature_name):
                '''
                Plot training and testing distributions of a feature
                ax: matplotlib axes instance
                feature_name: name of the feature distribution to plot
                '''
                logging.debug('Plotting feature: {0}'.format(feature_name))

                from dataprovider import PandasSurvivedClassSelector, PandasDrownedClassSelector
                data_provider = self.model.get_data_provider(self.model._configuration[feature_name]['data_provider'])
                class1_selector = self.model._configuration[feature_name]['class1']
                class2_selector  = self.model._configuration[feature_name]['class2']
        
                class1_training = np.array(data_provider.get_training_examples(feature_name,class1_selector).fillna(-1))
                class1_testing = np.array(data_provider.get_testing_examples(feature_name,class1_selector).fillna(-1))
                class2_training = np.array(data_provider.get_training_examples(feature_name,class2_selector).fillna(-1))
                class2_testing = np.array(data_provider.get_testing_examples(feature_name,class2_selector).fillna(-1))

                EPS = 0.0001
                class1_training = np.log10(class1_training+EPS)
                class1_testing = np.log10(class1_testing+EPS)
                class2_training = np.log10(class2_training+EPS)
                class2_testing = np.log10(class2_testing+EPS)
                
                # Binning configuration
                underflow, overflow = self.model._configuration[feature_name]['style']['under_over_flow']
                bins = self.model._configuration[feature_name]['style']['bins']
                bin_centers = bins[0:-1]+np.diff(bins)/2.
                logging.debug('bin_centers ({0})={{1}}'.format(feature_name,bin_centers))

                # Class 1 training and testing distributions
                if any([underflow, overflow]):
                        ax.hist(np.clip(class1_training, bins[0] if underflow else None, bins[-1] if overflow else None), bins,
                        density=True, histtype='stepfilled',
                        color=self.model._configuration[feature_name]['class1_color_train'], 
                        label=self.model._configuration[feature_name]['class1_label_train'], alpha = 0.5)
                else:
                        ax.hist(class1_training, bins, 
                        density=True, histtype='stepfilled', 
                        color=self.model._configuration[feature_name]['class1_color_train'], 
                        label=self.model._configuration[feature_name]['class1_label_train'], alpha = 0.5)
                hist_testing = np.histogram(class1_testing, bins)
                if any([underflow, overflow]):
                        hist_testing = np.histogram(np.clip(class1_testing, bins[0] if underflow else None, bins[-1] if overflow else None), bins)
                points_testing_y = hist_testing[0]/np.diff(bins)/float(np.sum(hist_testing[0]))
                points_testing_yerr = np.sqrt(hist_testing[0])/np.diff(bins)/float(np.sum(hist_testing[0]))
                ax.errorbar(bin_centers, points_testing_y, yerr=points_testing_yerr, marker=self.model._configuration[feature_name]['class1_marker_test'], 
                                ls=self.model._configuration[feature_name]['class1_line_test'], color=self.model._configuration[feature_name]['class1_color_test'], 
                                label=self.model._configuration[feature_name]['class1_label_test'])

                # Class 2 training and testing distributions
                if any([underflow, overflow]):
                        plt.rcParams['hatch.color'] = self.model._configuration[feature_name]['class2_color_train']
                        ax.hist(np.clip(class2_training, bins[0] if underflow else None, bins[-1] if overflow else None), bins,
                        density=True, histtype='stepfilled',
                        color=self.model._configuration[feature_name]['class2_color_train'], hatch='///',
                        label=self.model._configuration[feature_name]['class2_label_train'], alpha = 0.2)
                else:
                        plt.rcParams['hatch.color'] = self.model._configuration[feature_name]['class2_color_train']
                        ax.hist(class2_training, bins,
                        density=True, histtype='stepfilled',
                        color=self.model._configuration[feature_name]['class2_color_train'], hatch='///',
                        label=self.model._configuration[feature_name]['class2_label_train'], alpha = 0.2)
                        
                hist_testing = np.histogram(class2_testing, bins)
                if any([underflow, overflow]):
                        hist_testing = np.histogram(np.clip(class2_testing, bins[0] if underflow else None, bins[-1] if overflow else None), bins)
                points_testing_y = hist_testing[0]/np.diff(bins)/float(np.sum(hist_testing[0]))
                points_testing_yerr = np.sqrt(hist_testing[0])/np.diff(bins)/np.sum(hist_testing[0])
                ax.errorbar(bin_centers, points_testing_y, yerr=points_testing_yerr, marker=self.model._configuration[feature_name]['class2_marker_test'], 
                                ls=self.model._configuration[feature_name]['class2_line_test'], color=self.model._configuration[feature_name]['class2_color_test'], 
                                label=self.model._configuration[feature_name]['class2_label_test'])
                
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
                data_provider = self.model.get_data_provider(self.model._configuration[feature_name]['data_provider'])
                class1_selector = self.model._configuration[feature_name]['class1']
                class2_selector  = self.model._configuration[feature_name]['class2']
        
                class1_training = list(data_provider.get_training_examples(feature_name,class1_selector).fillna(-1))
                class1_testing = list(data_provider.get_testing_examples(feature_name,class1_selector).fillna(-1))
                class2_training = list(data_provider.get_training_examples(feature_name,class2_selector).fillna(-1))
                class2_testing = list(data_provider.get_testing_examples(feature_name,class2_selector).fillna(-1))
                
                # Binning configuration
                underflow, overflow = self.model._configuration[feature_name]['style']['under_over_flow']
                bins = self.model._configuration[feature_name]['style']['bins']
                bin_centers = bins[0:-1]+np.diff(bins)/2.
                logging.debug('bin_centers ({0})={{1}}'.format(feature_name,bin_centers))

                # Class 1 training and testing distributions
                if any([underflow, overflow]):
                        ax.hist(np.clip(class1_training, bins[0] if underflow else None, bins[-1] if overflow else None), bins,
                        density=True, histtype='stepfilled',
                        color=self.model._configuration[feature_name]['class1_color_train'], 
                        label=self.model._configuration[feature_name]['class1_label_train'], alpha = 0.5)
                else:
                        ax.hist(class1_training, bins, 
                        density=True, histtype='stepfilled', 
                        color=self.model._configuration[feature_name]['class1_color_train'], 
                        label=self.model._configuration[feature_name]['class1_label_train'], alpha = 0.5)
                hist_testing = np.histogram(class1_testing, bins)
                if any([underflow, overflow]):
                        hist_testing = np.histogram(np.clip(class1_testing, bins[0] if underflow else None, bins[-1] if overflow else None), bins)
                points_testing_y = hist_testing[0]/np.diff(bins)/float(np.sum(hist_testing[0]))
                points_testing_yerr = np.sqrt(hist_testing[0])/np.diff(bins)/float(np.sum(hist_testing[0]))
                ax.errorbar(bin_centers, points_testing_y, yerr=points_testing_yerr, marker=self.model._configuration[feature_name]['class1_marker_test'], 
                                ls=self.model._configuration[feature_name]['class1_line_test'], color=self.model._configuration[feature_name]['class1_color_test'], 
                                label=self.model._configuration[feature_name]['class1_label_test'])

                # Class 2 training and testing distributions
                if any([underflow, overflow]):
                        plt.rcParams['hatch.color'] = self.model._configuration[feature_name]['class2_color_train']
                        ax.hist(np.clip(class2_training, bins[0] if underflow else None, bins[-1] if overflow else None), bins,
                        density=True, histtype='stepfilled',
                        color=self.model._configuration[feature_name]['class2_color_train'], hatch='///',
                        label=self.model._configuration[feature_name]['class2_label_train'], alpha = 0.2)
                else:
                        plt.rcParams['hatch.color'] = self.model._configuration[feature_name]['class2_color_train']
                        ax.hist(class2_training, bins,
                        density=True, histtype='stepfilled',
                        color=self.model._configuration[feature_name]['class2_color_train'], hatch='///',
                        label=self.model._configuration[feature_name]['class2_label_train'], alpha = 0.2)
                        
                hist_testing = np.histogram(class2_testing, bins)
                if any([underflow, overflow]):
                        hist_testing = np.histogram(np.clip(class2_testing, bins[0] if underflow else None, bins[-1] if overflow else None), bins)
                points_testing_y = hist_testing[0]/np.diff(bins)/float(np.sum(hist_testing[0]))
                points_testing_yerr = np.sqrt(hist_testing[0])/np.diff(bins)/np.sum(hist_testing[0])
                ax.errorbar(bin_centers, points_testing_y, yerr=points_testing_yerr, marker=self.model._configuration[feature_name]['class2_marker_test'], 
                                ls=self.model._configuration[feature_name]['class2_line_test'], color=self.model._configuration[feature_name]['class2_color_test'], 
                                label=self.model._configuration[feature_name]['class2_label_test'])
                
                # Plot style
                self.style_feature_pad(ax,feature_name)

        @log_with()      
        def build_test_train_categorical_feature_pad(self, ax, feature_name):
                '''
                Plot training and testing distributions of a feature
                ax: matplotlib axes instance
                feature_name: name of the feature distribution to plot
                '''
                logging.debug('Plotting feature: {0}'.format(feature_name))

                from dataprovider import PandasSurvivedClassSelector, PandasDrownedClassSelector
                data_provider = self.model.get_data_provider(self.model._configuration[feature_name]['data_provider'])
                class1_selector = self.model._configuration[feature_name]['class1']
                class2_selector  = self.model._configuration[feature_name]['class2']
        
                class1_training = list(data_provider.get_training_examples(feature_name,class1_selector).fillna('-1'))
                class1_testing = list(data_provider.get_testing_examples(feature_name,class1_selector).fillna('-1'))
                class2_training = list(data_provider.get_training_examples(feature_name,class2_selector).fillna('-1'))
                class2_testing = list(data_provider.get_testing_examples(feature_name,class2_selector).fillna('-1'))
                
                # Binning configuration
                bins = self.model._configuration[feature_name]['style']['bins']

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
                ax.bar(bins, class1_training_values, color=self.model._configuration[feature_name]['class1_color_train'], 
                        label=self.model._configuration[feature_name]['class1_label_train'], alpha = 0.5)
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
                                marker=self.model._configuration[feature_name]['class1_marker_test'], 
                                ls=self.model._configuration[feature_name]['class1_line_test'],
                                color=self.model._configuration[feature_name]['class1_color_test'], 
                                label=self.model._configuration[feature_name]['class1_label_test'])

                # Class 2 training and testing distributions
                plt.rcParams['hatch.color'] = self.model._configuration[feature_name]['class2_color_train']
                ax.bar(bins, class2_training_values, color=self.model._configuration[feature_name]['class2_color_train'], 
                label=self.model._configuration[feature_name]['class2_label_train'], hatch='///', alpha = 0.5)
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
                                marker=self.model._configuration[feature_name]['class2_marker_test'], 
                                ls=self.model._configuration[feature_name]['class2_line_test'],
                                color=self.model._configuration[feature_name]['class2_color_test'], 
                                label=self.model._configuration[feature_name]['class2_label_test'])
                
                # Plot style
                self.style_feature_pad(ax,feature_name)

        @log_with()
        def draw(self):
                if not self.view_name: raise RuntimeError('Cannot build view. View name is not specified!')
                nrows=self.model._configuration[self.view_name]['layout']['nrows']
                ncols=self.model._configuration[self.view_name]['layout']['ncols']
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
                fig.set_size_inches(*self.model._configuration[self.view_name]['size'])
                pads = axes.flatten()

                for pad,feature in enumerate(self.model._configuration[self.view_name]['features']):
                        if self.model._configuration[feature]['style']['type'] == 'numerical':
                                self.build_test_train_numerical_feature_pad(pads[pad],feature)
                        elif self.model._configuration[feature]['style']['type'] == 'numerical_log_zoom':
                                self.build_test_train_numerical_zoom_feature_pad(pads[pad],feature)
                        elif self.model._configuration[feature]['style']['type'] == 'categorical':
                                self.build_test_train_categorical_feature_pad(pads[pad],feature)
                        else: 
                                logging.error('Unknown feature type: {}'.format(self.model._configuration[feature]['style']['type']))
                                raise NotImplementedError

                fig.tight_layout()
                self.set_outfilename(self.model._configuration[self.view_name]['output_filename'])
                for name in self.get_outfile_name(): plt.savefig(name)
                plt.close(fig)

class ViewModelRegressionLearningCurve(View):
        @log_with()
        def __init__(self,view_name=None):
                self.view_name = view_name
                pass
        
        @log_with()
        def draw(self):
                if not self.view_name: raise RuntimeError('Cannot build view. View name is not specified!')
                nrows=self.model._configuration[self.view_name]['layout']['nrows']
                ncols=self.model._configuration[self.view_name]['layout']['ncols']
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
                fig.set_size_inches(*self.model._configuration[self.view_name]['size'])
                pads = axes.flatten()
                for pad,metric in enumerate(self.model._configuration[self.view_name]['metrics']):
                        if metric in ['rmse','mae']:
                                epochs = len(self.model.fit_results['validation_0'][metric])
                                x_axis = range(0, epochs)
                                pads[pad].plot(x_axis, self.model.fit_results['validation_0'][metric], label='Train')
                                pads[pad].plot(x_axis, self.model.fit_results['validation_1'][metric], label='Test')
                                pads[pad].legend()
                                pads[pad].set_ylabel(metric)
                                pads[pad].set_xlabel('Num trees')
                                pads[pad].set_title('XGBoost {} Loss'.format(metric))
                        else: 
                                logging.error('Unknown metric type: {}'.format(metric))
                                raise NotImplementedError

                fig.tight_layout()
                # plt.show()
                self.set_outfilename(self.model._configuration[self.view_name]['output_filename'])
                for name in self.get_outfile_name(): plt.savefig(name)
                plt.close(fig)

class ViewModelRegressionLassoLarsIC(View):
        @log_with()
        def __init__(self,view_name=None):
                self.view_name = view_name
                pass
        
        @log_with()
        def draw(self):
                # This is to avoid division by zero while doing np.log10
                EPSILON = 1e-4
                if not self.view_name: raise RuntimeError('Cannot build view. View name is not specified!')
                nrows=self.model._configuration[self.view_name]['layout']['nrows']
                ncols=self.model._configuration[self.view_name]['layout']['ncols']
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
                fig.set_size_inches(*self.model._configuration[self.view_name]['size'])
                pads = axes.flatten()
                for pad,metric in enumerate(self.model._configuration[self.view_name]['metrics']):
                        if metric in ['aic','bic']:
                                alpha = self.model.fit_results[metric].alpha_ + EPSILON
                                alphas = self.model.fit_results[metric].alphas_ + EPSILON
                                criterion = self.model.fit_results[metric].criterion_
                                pads[pad].plot(-np.log10(alphas), criterion, '--', color='r', linewidth=3, label='{} criterion'.format(metric))
                                pads[pad].axvline(-np.log10(alpha), color='r', linewidth=3, label='alpha: {} estimate'.format(metric))
                                pads[pad].legend()
                                pads[pad].set_xlabel('-log(alpha)')
                                pads[pad].set_ylabel('criterion')
                                pads[pad].set_title('Information-criterion for model selection')
                        else: 
                                logging.error('Unknown metric type: {}'.format(metric))
                                raise NotImplementedError

                fig.tight_layout()
                # plt.show()
                self.set_outfilename(self.model._configuration[self.view_name]['output_filename'])
                for name in self.get_outfile_name(): plt.savefig(name)
                plt.close(fig)

class ViewModelMulticlassClassificationDistributions(View):
        @log_with()
        def __init__(self,view_name=None):
                self.view_name = view_name
                pass
        
        @log_with()
        def draw(self):
                if not self.view_name: raise RuntimeError('Cannot build view. View name is not specified!')
                nrows=self.model._configuration[self.view_name]['layout']['nrows']
                ncols=self.model._configuration[self.view_name]['layout']['ncols']
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
                fig.set_size_inches(*self.model._configuration[self.view_name]['size'])
                pads = axes.flatten()

                target_variable_names = self.model._configuration['model']['target']
                data_provider = self.model.get_data_provider(self.model._configuration['model']['data_provider'])

                input_features_names = self.model._configuration['model']['input_features']
                X_train = data_provider.train[input_features_names]
                y_train = data_provider.train[target_variable_names]

                X_test = data_provider.test[input_features_names]
                y_test = data_provider.test[target_variable_names]

                for pad,distribution in enumerate(self.model._configuration[self.view_name]['distributions']):
                        if distribution == 'classifier':
                                # Binning configuration
                                underflow, overflow = self.model._configuration[self.view_name]['style']['under_over_flow']
                                bins = self.model._configuration[self.view_name]['style']['bins']
                                bin_centers = bins[0:-1]+np.diff(bins)/2.
                                logging.debug('bin_centers ({0})={{1}}'.format(self.view_name,bin_centers))

                                y_pred_train = self.model.my_model.predict(X_train)
                                y_pred_test  = self.model.my_model.predict(X_test)

                                # Class 1 training and testing distributions
                                if any([underflow, overflow]):
                                        pads[pad].hist(np.clip(y_pred_train, bins[0] if underflow else None, bins[-1] if overflow else None), bins,
                                        density=True, histtype='stepfilled',
                                        color=self.model._configuration[self.view_name]['class1_color_train'], 
                                        label=self.model._configuration[self.view_name]['class1_label_train'], alpha = 0.5)
                                else:
                                        pads[pad].hist(y_pred_train, bins, 
                                        density=True, histtype='stepfilled', 
                                        color=self.model._configuration[self.view_name]['class1_color_train'], 
                                        label=self.model._configuration[self.view_name]['class1_label_train'], alpha = 0.5)
                                hist_testing = np.histogram(y_pred_test, bins)
                                if any([underflow, overflow]):
                                        hist_testing = np.histogram(np.clip(y_pred_test, bins[0] if underflow else None, bins[-1] if overflow else None), bins)
                                points_testing_y = hist_testing[0]/np.diff(bins)/float(np.sum(hist_testing[0]))
                                points_testing_yerr = np.sqrt(hist_testing[0])/np.diff(bins)/float(np.sum(hist_testing[0]))
                                pads[pad].errorbar(bin_centers, points_testing_y, yerr=points_testing_yerr, marker=self.model._configuration[self.view_name]['class1_marker_test'], 
                                ls=self.model._configuration[self.view_name]['class1_line_test'], color=self.model._configuration[self.view_name]['class1_color_test'], 
                                label=self.model._configuration[self.view_name]['class1_label_test'])

                                pads[pad].legend()
                                pads[pad].set_xlabel('Classifier output')
                                pads[pad].set_title('REJECT=0,MF=1,CC=2,CL=3'.format(distribution))
                        else: 
                                logging.error('Unknown distribution type: {}'.format(distribution))
                                raise NotImplementedError

                fig.tight_layout()
                # plt.show()
                self.set_outfilename(self.model._configuration[self.view_name]['output_filename'])
                for name in self.get_outfile_name(): plt.savefig(name)
                plt.close(fig)

class ViewModelMulticlassProbabilityCorrelations(View):
        @log_with()
        def __init__(self,view_name=None):
                self.view_name = view_name
                pass
        
        @log_with()
        def draw(self):
                if not self.view_name: raise RuntimeError('Cannot build view. View name is not specified!')
                nrows=self.model._configuration[self.view_name]['layout']['nrows']
                ncols=self.model._configuration[self.view_name]['layout']['ncols']
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
                fig.set_size_inches(*self.model._configuration[self.view_name]['size'])
                pads = axes.flatten()

                target_variable_names = self.model._configuration['model']['target']
                data_provider = self.model.get_data_provider(self.model._configuration['model']['data_provider'])

                input_features_names = self.model._configuration['model']['input_features']
                X_train = data_provider.train[input_features_names]
                y_train = data_provider.train[target_variable_names]

                X_test = data_provider.test[input_features_names]
                y_test = data_provider.test[target_variable_names]

                for pad,distribution in enumerate(self.model._configuration[self.view_name]['distributions']):
                        y_prob_train = self.model.my_model.predict_proba(X_train).reshape(len(X_train),4)
                        y_prob_test  = self.model.my_model.predict_proba(X_test).reshape(len(X_test),4)

                        feature_1, feature_2 = self.model._configuration[self.view_name][distribution]['features_id'][0]
                        class1_training_f1 = y_prob_train[::,feature_1]
                        class1_training_f2 = y_prob_train[::,feature_2]
                        pads[pad].scatter(class1_training_f1, class1_training_f2,
                                        color=self.model._configuration[self.view_name][distribution]['class0_color_train'],
                                        label=self.model._configuration[self.view_name][distribution]['class0_label_train'],
                                        marker=self.model._configuration[self.view_name][distribution]['class0_marker_train'], alpha = 0.5)

                        feature_1, feature_2 = self.model._configuration[self.view_name][distribution]['features_id'][1]
                        class1_training_f1 = y_prob_train[::,feature_1] 
                        class1_training_f2 = y_prob_train[::,feature_2] 
                        pads[pad].scatter(class1_training_f1, class1_training_f2,
                                        color=self.model._configuration[self.view_name][distribution]['class1_color_train'],
                                        label=self.model._configuration[self.view_name][distribution]['class1_label_train'],
                                        marker=self.model._configuration[self.view_name][distribution]['class1_marker_train'], alpha = 0.5)

                        feature_1, feature_2 = self.model._configuration[self.view_name][distribution]['features_id'][2]
                        class2_training_f1 = y_prob_train[::,feature_1]
                        class2_training_f2 = y_prob_train[::,feature_2]
                        pads[pad].scatter(class2_training_f1, class2_training_f2,
                                        color=self.model._configuration[self.view_name][distribution]['class2_color_train'],
                                        label=self.model._configuration[self.view_name][distribution]['class2_label_train'],
                                        marker=self.model._configuration[self.view_name][distribution]['class2_marker_train'], alpha = 0.5)

                        pads[pad].set_xlabel(self.model._configuration[self.view_name][distribution]['xlabel'])
                        pads[pad].set_ylabel(self.model._configuration[self.view_name][distribution]['ylabel'])
                        pads[pad].legend()
                        pass
                        # else: 
                        #         logging.error('Unknown distribution type: {}'.format(distribution))
                        #         raise NotImplementedError

                fig.tight_layout()
                # plt.show()
                self.set_outfilename(self.model._configuration[self.view_name]['output_filename'])
                for name in self.get_outfile_name(): plt.savefig(name)
                plt.close(fig)

class ViewModelROC(View):
        @log_with()
        def __init__(self,view_name=None):
                self.view_name = view_name
                pass
        
        @log_with()
        def draw(self):
                if not self.view_name: raise RuntimeError('Cannot build view. View name is not specified!')

                from sklearn.metrics import roc_curve, auc
                from sklearn.preprocessing import label_binarize
                from scipy import interp

                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()

                target_variable_names = self.model._configuration['model']['target']
                data_provider = self.model.get_data_provider(self.model._configuration['model']['data_provider'])
                input_features_names = self.model._configuration['model']['input_features']

                y       = data_provider.data[target_variable_names]
                X_train = data_provider.train[input_features_names]
                y_train = data_provider.train[target_variable_names]
                X_test = data_provider.test[input_features_names]
                y_test = data_provider.test[target_variable_names]
                y_score= self.model.fit_results.decision_function(X_test)

                # Binarize the output
                y = label_binarize(y, classes=[0, 1, 2, 3])
                n_classes = y.shape[1]

                for i in range(n_classes):
                        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])

                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

                # First aggregate all false positive rates
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

                # Then interpolate all ROC curves at this points
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(n_classes):
                        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

                # Finally average it and compute AUC
                mean_tpr /= n_classes

                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

                # Plot all ROC curves
                plt.figure()
                plt.plot(fpr["micro"], tpr["micro"],
                        label='micro-average ROC curve (area = {0:0.2f})'
                        ''.format(roc_auc["micro"]),
                        color='deeppink', linestyle=':', linewidth=4)

                plt.plot(fpr["macro"], tpr["macro"],
                        label='macro-average ROC curve (area = {0:0.2f})'
                        ''.format(roc_auc["macro"]),
                        color='navy', linestyle=':', linewidth=4)

                colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
                for i, color in zip(range(n_classes), colors):
                        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                        label='ROC curve of class {0} (area = {1:0.2f})'
                        ''.format(i, roc_auc[i]))

                plt.plot([0, 1], [0, 1], 'k--', lw=lw)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Some extension of Receiver operating characteristic to multi-class')
                plt.legend(loc="lower right")
                # plt.show()
                self.set_outfilename(self.model._configuration[self.view_name]['output_filename'])
                for name in self.get_outfile_name(): plt.savefig(name)
                plt.close(fig)

class ViewModelConfusionMatrix(View):
        @log_with()
        def __init__(self,view_name=None):
                self.view_name = view_name
                pass
        
        @log_with()
        def draw(self):
                if not self.view_name: raise RuntimeError('Cannot build view. View name is not specified!')
                nrows=self.model._configuration[self.view_name]['layout']['nrows']
                ncols=self.model._configuration[self.view_name]['layout']['ncols']
                # fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
                fig, axes = plt.subplots()
                fig.set_size_inches(*self.model._configuration[self.view_name]['size'])
                # pads = axes.flatten()

                target_variable_names = self.model._configuration['model']['target']
                data_provider = self.model.get_data_provider(self.model._configuration['model']['data_provider'])

                input_features_names = self.model._configuration['model']['input_features']
                X_train = data_provider.train[input_features_names]
                y_train = data_provider.train[target_variable_names]

                X_test = data_provider.test[input_features_names]
                y_test = data_provider.test[target_variable_names]
                y_pred = self.model.my_model.predict(X_test)

                normalize = True
                cmap=plt.cm.Blues
                titles_options = [("Confusion matrix, without normalization", None),
                                  ("Normalized confusion matrix", 'true')]
                from sklearn.metrics import confusion_matrix
                from sklearn.utils.multiclass import unique_labels
                # Compute confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                # Only use the labels that appear in the data
                classes = [0,1,2,3]
                # classes = classes[unique_labels(y_test, y_pred)]
                if normalize:
                        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                        print("Normalized confusion matrix")
                else:
                        print('Confusion matrix, without normalization')

                print cm

                im = axes.imshow(cm, interpolation='nearest', cmap=cmap)
                axes.figure.colorbar(im, ax=axes)
                # We want to show all ticks...
                axes.set(xticks=np.arange(cm.shape[1]),
                        yticks=np.arange(cm.shape[0]),
                        # ... and label them with the respective list entries
                        xticklabels=classes, yticklabels=classes,
                        title=titles_options[1][0],
                        ylabel='True label',
                        xlabel='Predicted label')

                # Rotate the tick labels and set their alignment.
                plt.setp(axes.get_xticklabels(), rotation=45, ha="right",
                        rotation_mode="anchor")

                # Loop over data dimensions and create text annotations.
                fmt = '.2f' if normalize else 'd'
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                                axes.text(j, i, format(cm[i, j], fmt),
                                        ha="center", va="center",
                                        color="white" if cm[i, j] > thresh else "black")
                # fig.tight_layout()
                # return ax

                        # pads[0] = disp.ax_
                        # pads[pad].set_xlabel(self.model._configuration[self.view_name][distribution]['xlabel'])
                        # pads[pad].set_ylabel(self.model._configuration[self.view_name][distribution]['ylabel'])


                        pass
                        # else: 
                        #         logging.error('Unknown distribution type: {}'.format(distribution))
                        #         raise NotImplementedError

                fig.tight_layout()
                # plt.show()
                self.set_outfilename(self.model._configuration[self.view_name]['output_filename'])
                for name in self.get_outfile_name(): plt.savefig(name)
                plt.close(fig)

class ViewModelClassificationLearningCurve(View):
        @log_with()
        def __init__(self,view_name=None):
                self.view_name = view_name
                pass
        
        @log_with()
        def draw(self):
                if not self.view_name: raise RuntimeError('Cannot build view. View name is not specified!')
                nrows=self.model._configuration[self.view_name]['layout']['nrows']
                ncols=self.model._configuration[self.view_name]['layout']['ncols']
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
                fig.set_size_inches(*self.model._configuration[self.view_name]['size'])
                pads = axes.flatten()
                for pad,metric in enumerate(self.model._configuration[self.view_name]['metrics']):
                        if metric in ["error", "auc", "map"]:
                                epochs = len(self.model.fit_results['validation_0'][metric])
                                x_axis = range(0, epochs)
                                pads[pad].plot(x_axis, self.model.fit_results['validation_0'][metric], label='Train')
                                pads[pad].plot(x_axis, self.model.fit_results['validation_1'][metric], label='Test')
                                pads[pad].legend()
                                pads[pad].set_ylabel(metric)
                                pads[pad].set_xlabel('Num trees')
                                pads[pad].set_title('XGBoost {} Loss'.format(metric))
                        else: 
                                logging.error('Unknown metric type: {}'.format(metric))
                                raise NotImplementedError

                fig.tight_layout()
                # plt.show()
                self.set_outfilename(self.model._configuration[self.view_name]['output_filename'])
                for name in self.get_outfile_name(): plt.savefig(name)
                plt.close(fig)

class ViewModelMulticlassClassificationLearningCurve(View):
        @log_with()
        def __init__(self,view_name=None):
                self.view_name = view_name
                pass
        
        @log_with()
        def draw(self):
                if not self.view_name: raise RuntimeError('Cannot build view. View name is not specified!')
                nrows=self.model._configuration[self.view_name]['layout']['nrows']
                ncols=self.model._configuration[self.view_name]['layout']['ncols']
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
                fig.set_size_inches(*self.model._configuration[self.view_name]['size'])
                pads = axes.flatten()
                for pad,metric in enumerate(self.model._configuration[self.view_name]['metrics']):
                        if metric in ["merror", "mlogloss"]:
                                epochs = len(self.model.fit_results['validation_0'][metric])
                                x_axis = range(0, epochs)
                                pads[pad].plot(x_axis, self.model.fit_results['validation_0'][metric], label='Train')
                                pads[pad].plot(x_axis, self.model.fit_results['validation_1'][metric], label='Test')
                                pads[pad].legend()
                                pads[pad].set_ylabel(metric)
                                pads[pad].set_xlabel('Num trees')
                                pads[pad].set_title('XGBoost {} Loss'.format(metric))
                        else: 
                                logging.error('Unknown metric type: {}'.format(metric))
                                raise NotImplementedError

                fig.tight_layout()
                # plt.show()
                self.set_outfilename(self.model._configuration[self.view_name]['output_filename'])
                for name in self.get_outfile_name(): plt.savefig(name)
                plt.close(fig)

class LatexBeamerView(View):
        @log_with()
        def __init__(self,view_name=None):
                self.view_name=view_name
                self.views = set()
                pass

        @log_with()
        def add_view(self, view):
                self.views.add(view)
                pass

        @log_with()
        def Init(self):
                pass

        @log_with()
        def save(self,serializer):
                for view in self.views: view.save(serializer)
                serializer.serialize_beamer_view(self)

        @log_with()
        def annotate(self,type):
                if type == "screen":
                        bright_green_text = "\033[1;32;40m"
                        normal_text = "\033[0;37;40m"
                        print "\n".join(textwrap.wrap(bcolors.OKBLUE+
                                                self.model._annotation.encode('ascii')+
                                                bcolors.ENDC, 120))
                elif type == "tex":
                        logging.warning("Annotation format: {}. Not implemented yet!".format(type))
                elif type == "md":
                        logging.warning("Annotation format: {}. Not implemented yet!".format(type))
                else:
                        logging.error("Annotation format not recognized: {}".format(type))

        @log_with()
        def draw(self):
                self.Init()
		for view in self.views: view.draw()
                logging.debug( pp.pformat(self.model._configuration) )
                subprocess.call(["pdflatex", "-interaction=nonstopmode", "-output-directory={}".format(self._outputfolder), 
                                 self.model._configuration['latex_main']])

class LatexReportView(View):
        @log_with()
        def __init__(self,view_name=None):
                self.view_name=view_name
                self.views = set()
                pass

        @log_with()
        def add_view(self, view):
                self.views.add(view)
                pass

        @log_with()
        def Init(self):
                pass


        @log_with()
        def save(self,serializer):
                for view in self.views: view.save(serializer)
                serializer.serialize_report_view(self)

        @log_with()
        def annotate(self,type):
                if type == "screen":
                        bright_green_text = "\033[1;32;40m"
                        normal_text = "\033[0;37;40m"
                        print "\n".join(textwrap.wrap(bcolors.OKBLUE+
                                                self.model._annotation.encode('ascii')+
                                                bcolors.ENDC, 120))
                elif type == "tex":
                        logging.warning("Annotation format: {}. Not implemented yet!".format(type))
                elif type == "md":
                        logging.warning("Annotation format: {}. Not implemented yet!".format(type))
                else:
                        logging.error("Annotation format not recognized: {}".format(type))
        
        @log_with()
        def draw(self):
                self.Init()
		for view in self.views: view.draw()
                logging.debug( pp.pformat(self.model._configuration) )
                # subprocess.call(["pdflatex", "-interaction=nonstopmode", "-output-directory={}".format(self._outputfolder),
                                #  self.model._configuration['latex_main']])
