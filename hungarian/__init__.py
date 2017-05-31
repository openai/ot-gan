import os

import numpy
import pyximport

pyximport.install(setup_args=dict(
    include_dirs=[
        os.path.dirname(os.path.realpath(__file__)),
        numpy.get_include()
    ],
    extra_compile_args=['-O3', '-march=native'],
))
from .cyhungarian import match

del numpy, os, pyximport
