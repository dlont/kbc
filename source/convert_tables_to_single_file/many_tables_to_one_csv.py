#!/usr/bin/env python

"""
Convert multiple tables into single .csv file.
"""
__version__ = "1.0pre"

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

from logwith import *

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def main(args):
    '''Program algorithm entry point'''

    

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
