import os
import shutil

from numpy import loadtxt
from numpy import asarray, array

class StreamingDemo(object):

    def __init__(self, npoints=50, nbatches=5):
        self.npoints = npoints
        self.nbatches = nbatches

    def params(*args, **kwargs):
        """ Get analysis-specific parameters """
        raise NotImplementedError

    def run(*args, **kwargs):
        """ Run the streaming demo """
        raise NotImplementedError

    @staticmethod
    def make(demoname, *args, **kwargs):
        """ Create a streaming demo """

        from mlstreaming.kmeans import StreamingKMeans

        DEMOS = {
            'kmeans': StreamingKMeans
        }
        return DEMOS[demoname](*args, **kwargs)

    def setup(self, path, overwrite=False):
        """ Setup paths for a streaming demo where temporary data will be read / written"""

        if os.path.isdir(path):
            if overwrite:
                shutil.rmtree(path)
                os.mkdir(path)
            else:
                raise Exception('Base directory %s already exists and overwrite is set to False' % path)
        else:
            os.mkdir(path)

        datain = os.path.join(path, 'input')
        datainlabels = os.path.join(path, 'inputlbl')
        dataout = os.path.join(path, 'output')
        if os.path.isdir(datain):
            shutil.rmtree(datain)
        if os.path.isdir(datainlabels):
            shutil.rmtree(datainlabels)
        if os.path.isdir(dataout):
            shutil.rmtree(dataout)
        os.mkdir(datain)
        os.mkdir(datainlabels)
        os.mkdir(dataout)
        self.datain = datain
        self.datainlabels = datainlabels
        self.dataout = dataout
        return self

    def writepoints(self, pts, i):
        """ Write data points in a form that can be read by MLlib's vector parser  """

        f = file(os.path.join(self.datain, 'batch%g.txt' % i), 'w')
        s = map(lambda p: ",".join(str(p).split()).replace('[,', '[').replace(',]', ']'), pts)
        tmp = "\n".join(s)
        f.write(tmp)
        f.close()

    def writecenters(self, centers, i):
        """ Write integer labels cluster membership from make_blobs """

        # This needs to be written to a separate directory as StreamingContext.textFileStream
        # takes only a directory without any means to filter out files
        f = file(os.path.join(self.datainlabels, 'labels%g.txt' % i), 'w')
        s = map(lambda p: " ".join(str(p).split()).replace('[', '').replace(']', ''), centers)
        tmp = " ".join(s)
        f.write(tmp)
        f.close()

    # Helper function when the Lightning client is separated (eg. IPython)
    def readcenters(self, i):
        """ Read integer labels """

        try:
            centers = loadtxt(os.path.join(datainlabels, 'labels%g.txt' % i), dtype=int)
        except:
            print('Cannot load cluster membership labels')
            return array([])

        return centers

    # Helper function when the Lightning client is separated (eg. IPython)
    def readpts(i):
        """ Read original points """

        try:
            with open (os.path.join(datain, 'batch%g.txt' % i), 'r') as ptsfile:
                ptscontent=ptsfile.read().replace('[', '').replace(']', '')

            pts = loadtxt(StringIO(ptscontent), delimiter=',')
        except:
            print('Cannot load points', sys.exc_info())
            return array([])

        return pts
