import logging
import multiprocessing
import os
import random

import h5py
# not used in DeepEthogram; only to easily show plots
from IPython.display import Image
from omegaconf import OmegaConf
import pandas as pd
import torch

from deepethogram import configuration, postprocessing, projects, utils
from deepethogram.debug import print_dataset_info
from deepethogram.flow_generator.train import flow_generator_train
from deepethogram.feature_extractor.train import feature_extractor_train
from deepethogram.feature_extractor.inference import feature_extractor_inference
from deepethogram.sequence.train import sequence_train
from deepethogram.sequence.inference import sequence_inference




project_path = '/n/data1/hms/neurobio/sabatini/Janet/Kim_segmented_deepethogram' #EDIT HERE
files = os.listdir(project_path)
assert 'DATA' in files, 'DATA directory not found! {}'.format(files)
assert 'models' in files, 'models directory not found! {}'.format(files)
assert 'project_config.yaml' in files, 'project config not found! {}'.format(files)

preset = 'deg_f'

def reset_logger():
  # First, overwrite any logger so that we can actually see log statements
  # https://stackoverflow.com/questions/13839554/how-to-change-filehandle-with-python-logging-on-the-fly-with-different-classes-a
  log = logging.getLogger()  # root logger
  log.setLevel(logging.INFO)
  for hdlr in log.handlers[:]:  # remove all old handlers
      log.removeHandler(hdlr)
  log.addHandler(logging.StreamHandler())
  return log

log = reset_logger()


print("Processing sequence inference")

cfg = configuration.make_sequence_inference_cfg(project_path)
cfg.sequence.weights = 'latest'
cfg.compute.num_workers = 8 #n_cpus  EDIT HERE
#cfg.inference.directory_list = ['/n/data1/hms/neurobio/sabatini/Janet/Kim_segmented_deepethogram/DATA/Unlabeled_001',
#                           '/n/data1/hms/neurobio/sabatini/Janet/Kim_segmented_deepethogram/DATA/Unlabeled_002',
#                           '/n/data1/hms/neurobio/sabatini/Janet/Kim_segmented_deepethogram/DATA/Unlabeled_003']
cfg.inference.overwrite = True
cfg.inference.ignore_error = False

sequence_inference(cfg)

# this just parses our DATA directory, to get the path to each file for each video
records = projects.get_records_from_datadir(os.path.join(project_path, 'DATA'))
animal = random.choice(list(records.keys()))
record = records[animal]

# I call the file output by inference the `outputfile` in various places in the code
outputfile = record['output']

utils.print_hdf5(outputfile)

# we use the h5py package for this
with h5py.File(outputfile, 'r') as f:
  probabilities = f['tgmj/P'][:]
  thresholds = f['tgmj/thresholds'][:]
# n frames x K behaviors
print(probabilities.shape)
print(thresholds)

cfg = configuration.make_postprocessing_cfg(project_path=project_path)

postprocessing.postprocess_and_save(cfg)

# Look at a record to see what's in it
#load a random record
records = projects.get_records_from_datadir(os.path.join(project_path, 'DATA'))
animal = random.choice(list(records.keys()))
record = records[animal]
# figure out the filename
predictions_filename = os.path.join(os.path.dirname(record['rgb']), record['key'] + '_predictions.csv')
assert os.path.isfile(predictions_filename)

# read csv
df = pd.read_csv(predictions_filename, index_col=0)
# display output
print(predictions_filename)
df.head()

print("Finished sequence inference")