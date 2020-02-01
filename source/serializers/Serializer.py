import shutil
import glob

from logwith import *
from views import *


class Serializer(object):
        @log_with()
        def __init__(self,builddir='build'):
                self._buildfolder = builddir
                self._outputfolder = None
                pass
        
        @log_with()
        def set_outputfolder(self,folder):
                self._outputfolder = folder
                if not os.path.exists(folder):
                        os.makedirs(folder)

        @log_with()
        def move_builddir_to_outputfolder(self):
                print(self._buildfolder, self._outputfolder, (self._buildfolder and self._outputfolder))
                if self._buildfolder is not None and self._outputfolder is not None:
                        for extension in ['pdf','png','tex']:
                                for file in glob.glob('{}/*.{}'.format(self._buildfolder,extension)):
                                        shutil.move(file, self._outputfolder)

        @log_with()
        def serialize_view(self,View):
                self.move_builddir_to_outputfolder()
                pass
        
        @log_with()
        def serialize_beamer_view(self,View):
                self.move_builddir_to_outputfolder()
                pass

        
        @log_with()
        def serialize_report_view(self,View):
                self.move_builddir_to_outputfolder()
                pass
		