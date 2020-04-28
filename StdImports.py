"""Standard imports"""
from Data import Loader,get_kind,get_neuro,Datas,rootDir
from Data import Data
from Data import gpData
from Tools import *
from Plotting import *

import re
import platform
import copy
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
if platform.system() is not 'Windows':
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import matplotlib.colors as colors


import seaborn as sns; sns.set()

ids = pd.IndexSlice # for multi-indexing

rand_state=198

loader=Loader()

dark_jet_r = cmap_map(lambda x: x*.9, matplotlib.cm.jet_r)


np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)



odor_conc_regex = '(?P<odor>\d?\w+\d?\w+)(?P<conc>\d{2})'

odor_color_dict = {odor:sns.color_palette("Set2")[i] for i, odor in enumerate(['EA','EB','MH','1o3o','Bzald','Acet'])}













