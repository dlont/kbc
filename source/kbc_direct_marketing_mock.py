#!/usr/bin/env python

"""
KBC test modelling task.
"""
__version__ = "1.2pre"

import os
import sys
import imp
import time
import glob
import shutil
import argparse
import subprocess
import logging
import json

import ROOT as rt

# My imports
from models import *
from views import *
from serializers import *
from logwith import *


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def progress(current, total, status=''):
        fullBarLength = 80
        doneBarLength = int(round(fullBarLength * current / float(total)))

        percents = round(100.0 * current / float(total), 1)
        bar = '>' * doneBarLength + ' ' * (fullBarLength - doneBarLength)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()

class Style(object):
        def __init__(self, config_json, model):
                self._json = config_json
                self._model = model
        model = property(None,None)

        @log_with()
        def decorate(self,obj):
                """
                Decorate object of the model.
                Assumes Drawable object from ROOT
                """
                name = obj.GetName()
                return obj

        @log_with()
        def decorate_stack(self, stack):
                pass

        @log_with()
        def decorate_graph(self,mg):
                pass

        @log_with()
        def make_legend(self,c,objlist,**kwargs):
                header=kwargs.get('header',None)
                pos=kwargs.get('pos',(0.11,0.11,0.5,0.5))
                legend = rt.TLegend(*pos)
                legend.SetName("TLeg_"+c.GetName())
                rt.SetOwnership(legend,False)
                legend.SetBorderSize(0)
                if header: legend.SetHeader(header)
                for el in objlist:
                        pass
                        # legend.AddEntry(self._model.get(el),self._json[el]['legend']['name'],self._json[el]['legend']['style'])
                c.cd()
                legend.Draw()

        @log_with()
        def decorate_pad(self, pad):
                pad.SetBottomMargin(0.2)
                pad.SetLeftMargin(0.2)
                pad.SetRightMargin(0.05)
                # pad.SetLogy()
                pad.Update()

        @log_with()
        def decorate_canvas(self, canvas):
                canvas.SetLeftMargin(1.1)
                canvas.SetRightMargin(0.1)
                # canvas.SetLogy()
                canvas.Update()

def main(arguments):

        #Load configuration .json file
        for config in arguments.config:
            configuration = None
            if ".json" in config:
                with open(config) as json_data:
                        configuration = json.load(json_data)
                        logging.debug(pp.pformat(configuration))
            elif "_cff.py" in config:
                    module_name = 'module_'+config.strip('.py')
                    configuration_module = imp.load_source(module_name, config)
                    configuration = configuration_module.config
            logging.debug(pp.pformat(configuration))

            model = None
            if configuration['model']['type'] == 'vanilla_regression': model = VanillaModelRegression(configuration)
            elif configuration['model']['type'] == 'vanilla_LassoLarsIC_regression': model = VanillaModelLassoLarsIC(configuration)
            elif configuration['model']['type'] == 'vanilla_classification': model = VanillaModelClassification(configuration)
            elif configuration['model']['type'] == 'advanced_classification': model = AdvancedModelClassification(configuration)
            elif configuration['model']['type'] == 'advanced_regression': model = AdvancedModelRegression(configuration)
            else: raise NotImplementedError
            
            style = Style(configuration,model)

            document = None
            if configuration['mode'] == 'beamer':
                    print "beamer option is not implemented!"
                    document = LatexBeamerView()
            elif configuration['mode'] == 'report':
                    print "report option is not implemented!"
                    document = LatexReportView()
            else: raise NotImplementedError
            document.set_model(model)
            document.set_style(style)
            document.set_builddir(arguments.builddir)
            # document.set_outfilename(arguments.outfile)
            # document.set_extension(arguments.extension)
            serializer = Serializer(builddir=arguments.builddir)
            serializer.set_outputfolder(arguments.dir)
            if arguments.annotation_format: document.annotate(arguments.annotation_format)
            
            view = None
            for view_name in configuration['views']:
                    if configuration[view_name]['type'] == '1d_train_test':
                            view = View1dTrainTest(view_name)
                    elif configuration[view_name]['type'] == '2d_train_correlations':
                            view = View2dCorrelationsTrain(view_name)
                    elif configuration[view_name]['type'] == 'regression_model_learning_curve':
                            view = ViewModelRegressionLearningCurve(view_name)
                    elif configuration[view_name]['type'] == 'regression_model_lasso_lars_ic':
                            view = ViewModelRegressionLassoLarsIC(view_name)
                    elif configuration[view_name]['type'] == 'classification_model_learning_curve':
                            view = ViewModelClassificationLearningCurve(view_name)
                    elif configuration[view_name]['type'] == 'multiclassification_model_learning_curve':
                            view = ViewModelMulticlassClassificationLearningCurve(view_name)   
                    else: view = View(view_name)
                    view.set_model(model)
                    view.set_style(style)
                    view.set_builddir(arguments.builddir)
                    view.set_outfilename(arguments.outfile)
                    view.set_extension(arguments.extension)
                    if arguments.annotation_format: view.annotate(arguments.annotation_format)
                    document.add_view(view)

            document.draw()

            document.save(serializer)
            configuration['command']=' '.join(sys.argv)
            document.save_config(config)


        return 0


if __name__ == '__main__':
        start_time = time.time()

        parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)
        parser.add_argument('-o', '--outfile', help="Output file", default='test')
        parser.add_argument('-e', '--extension', help="Plot file extension (.C, .root, .png, .pdf)", default='png')
        parser.add_argument('--builddir', help="Build directory", default='build')
        parser.add_argument('--dir', help="Result output directory", default='.')
        parser.add_argument('-c', '--config', help=".json or _cff.py configuration file", required=True, nargs='+')
        parser.add_argument('-a', '--annotation_format', default="screen",\
                            help="Print annotation in given format (screen, tex, md)")
        parser.add_argument('--no-annotation', dest='annotation_format', action='store_false',\
                                                help="Disable annotating")
        parser.add_argument('-b', help="ROOT batch mode", dest='isBatch', action='store_true')
        parser.add_argument(
                        '-d', '--debug',
                        help="Print lots of debugging statements",
                        action="store_const", dest="loglevel", const=logging.DEBUG,
                        default=logging.WARNING,
                        )
        parser.add_argument(
                        '-v', '--verbose',
                        help="Be verbose",
                        action="store_const", dest="loglevel", const=logging.INFO,
                        )

        args = parser.parse_args(sys.argv[1:])

        print(args)

        logging.basicConfig(level=args.loglevel)

        logging.info( time.asctime() )
        exitcode = main(args)
        logging.info( time.asctime() )
        logging.info( 'TOTAL TIME IN MINUTES:' + str((time.time() - start_time) / 60.0))
        sys.exit(exitcode)
