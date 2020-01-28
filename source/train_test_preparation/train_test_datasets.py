#!/usr/bin/env python

"""
Convert multiple tables into single .csv file.
Create a single file with entire datasets. Train and test samples can be obtained in a few lines of code in the modelling script.
"""
__version__ = "1.1"

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
import pandas as pd
import pprint as pp

from logwith import *

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def main(arguments):
    '''Program algorithm entry point'''

    from sklearn.model_selection import train_test_split

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

    df_Soc_Dem  = pd.read_csv(configuration['Soc_Dem_file'],index_col='Client')
    df_In_Out   = pd.read_csv(configuration['Inflow_Outflow_file'],index_col='Client')
    df_Products = pd.read_csv(configuration['Products_ActBalance_file'],index_col='Client')
    df_Sales    = pd.read_csv(configuration['Sales_Revenues_file'],index_col='Client')

    # Join different tables
    df_final    = df_Soc_Dem.join([df_In_Out,df_Products,df_Sales])

    # Binary encoding of textual genedel labels
    df_final['Sex'] = df_final['Sex'].replace({'M':0,'F':1})
    df_final = df_final.astype({'Sex':'int32'})

    print df_final.sort_values(by='Client').head()

    # Output entire dataset to file
    df_final.to_csv(configuration['output_file']['file'])


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
