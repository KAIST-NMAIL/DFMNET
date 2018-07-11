import errno
import os

def makeFolder(location):
    if not os.path.exists(location):
        try:
            os.makedirs(location)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    return 0