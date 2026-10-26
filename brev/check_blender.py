import blenderproc as bproc  # must stay the first import - BlenderProc re-launches this file inside Blender

# Run once by setup.sh before any rendering. The first `blenderproc run` on a machine
# installs BlenderProc's own Python packages into Blender; if eight render jobs make
# that first run at the same time, their pip installs overwrite each other and leave
# numpy/scipy half-installed ("numpy.core.multiarray failed to import"). Doing it
# here, alone, avoids that - and the imports below prove the result works.

import cv2
import numpy
import scipy.sparse.linalg  # noqa: F401 - the module the race broke first
import yaml  # noqa: F401

print(f"[check] numpy {numpy.__version__}, opencv {cv2.__version__}, scipy {scipy.__version__} - OK")
