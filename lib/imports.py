import collections
import datetime
import itertools
import json
import math
import os
import pickle
import random
import re
import string
import sys
import threading
import time
import warnings
from abc import abstractmethod
from collections import Counter, Iterable, OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from glob import glob, iglob
from itertools import chain
from operator import attrgetter, itemgetter

import bcolz
import cv2
import graphviz
import IPython
import matplotlib
import numpy as np
import pandas as pd
import PIL
import scipy
import seaborn as sns
import sklearn
import sklearn_pandas
from IPython.lib.deepreload import reload as dreload
from IPython.lib.display import FileLink
from ipywidgets import fixed, interact, interactive, widgets
from isoweek import Week
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib import rcParams
from pandas_summary import DataFrameSummary
from PIL import Image, ImageEnhance, ImageOps
from sklearn import ensemble, metrics, preprocessing

matplotlib.rc("animation", html="html5")
np.set_printoptions(precision=5, linewidth=110, suppress=True)


def in_notebook():
    return "ipykernel" in sys.modules


import tqdm as tq
from tqdm import tnrange, tqdm_notebook


def clear_tqdm():
    inst = getattr(tq.tqdm, "_instances", None)
    if not inst:
        return
    for i in range(len(inst)):
        inst.pop().close()


if in_notebook():

    def tqdm(*args, **kwargs):
        clear_tqdm()
        return tq.tqdm(*args, file=sys.stdout, **kwargs)

    def trange(*args, **kwargs):
        clear_tqdm()
        return tq.trange(*args, file=sys.stdout, **kwargs)


else:
    from tqdm import tqdm, trange

    tnrange = trange
    tqdm_notebook = tqdm

