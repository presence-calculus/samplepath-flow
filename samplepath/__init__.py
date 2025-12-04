# -*- coding: utf-8 -*-
from importlib.metadata import PackageNotFoundError, version  # Py≥3.8

try:
    __version__ = version("samplepath")  # use your *distribution* name
except PackageNotFoundError:
    __version__ = "0+unknown"
