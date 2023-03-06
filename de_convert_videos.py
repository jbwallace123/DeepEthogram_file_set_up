from deepethogram import projects
import os
import glob
import pandas as pd
import numpy as np

from deepethogram.projects import convert_all_videos

print("Imported deepethogram projects")

convert_all_videos('/n/data1/hms/neurobio/sabatini/Janet/Kim_segmented_deepethogram/project_config.yaml', movie_format='hdf5')