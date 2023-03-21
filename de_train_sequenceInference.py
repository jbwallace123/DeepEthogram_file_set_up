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

print("Imported deepethogram sequence inference")

project_path = '/n/data1/hms/neurobio/sabatini/Janet/Kim_segmented_deepethogram' #EDIT HERE
files = os.listdir(project_path)
assert 'DATA' in files, 'DATA directory not found! {}'.format(files)
assert 'models' in files, 'models directory not found! {}'.format(files)
assert 'project_config.yaml' in files, 'project config not found! {}'.format(files)

cfg = configuration.make_sequence_train_cfg(project_path=project_path)
cfg.compute.num_workers = 8 #EDIT HERE

sequence_model = sequence_train(cfg)

model_path = os.path.join(project_path, 'models')
weights = projects.get_weights_from_model_path(model_path)
sequence_weights = weights['sequence']
# bthe sequence type is always tgmj, a slightly modified TGM model
latest_weights = sequence_weights['tgmj'][-1]
# our run directory is two steps above the weight file
run_dir = os.path.dirname(os.path.dirname(latest_weights))
assert os.path.isdir(run_dir), 'run directory not found! {}'.format(run_directory)

figure_dir = os.path.join(run_dir, 'figures')
figure_files = utils.get_subfiles(figure_dir, 'file')
assert len(figure_files) >= 1

#Image(figure_files[0])

cfg = configuration.make_sequence_inference_cfg(project_path)
cfg.sequence.weights = 'latest'
cfg.compute.num_workers = n_cpus
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