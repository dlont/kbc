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
        def save_config(self, config):
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
        def draw(self):
                if not self.view_name: raise RuntimeError('Cannot build view. View name is not specified!')
                nrows=self.model._configuration[self.view_name]['layout']['nrows']
                ncols=self.model._configuration[self.view_name]['layout']['ncols']
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
                fig.set_size_inches(*self.model._configuration[self.view_name]['size'])
                pads = axes.flatten()

                for pad,features in enumerate(self.model._configuration[self.view_name]['features']):
                        if True:
                                self.model.build_train_correlation_pad(pads[pad],tuple(features))
                        else: 
                                logging.error('Unknown feature type: {}'.format(self.model._configuration[feature]['style']['type']))
                                raise NotImplementedError

                fig.tight_layout()
                self.set_outfilename(self.model._configuration[self.view_name]['output_filename'])
                for name in self.get_outfile_name(): plt.savefig(name)

class View1dTrainTest(View):
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

                for pad,feature in enumerate(self.model._configuration[self.view_name]['features']):
                        if self.model._configuration[feature]['style']['type'] == 'numerical':
                                self.model.build_test_train_numerical_feature_pad(pads[pad],feature)
                        elif self.model._configuration[feature]['style']['type'] == 'categorical':
                                self.model.build_test_train_categorical_feature_pad(pads[pad],feature)
                        else: 
                                logging.error('Unknown feature type: {}'.format(self.model._configuration[feature]['style']['type']))
                                raise NotImplementedError

                fig.tight_layout()
                self.set_outfilename(self.model._configuration[self.view_name]['output_filename'])
                for name in self.get_outfile_name(): plt.savefig(name)

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
