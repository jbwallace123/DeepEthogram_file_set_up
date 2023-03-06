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

print("Imported deepethogram feature extractor inference")

project_path = '/n/data1/hms/neurobio/sabatini/Janet/Kim_segmented_deepethogram' #EDIT HERE
files = os.listdir(project_path)
assert 'DATA' in files, 'DATA directory not found! {}'.format(files)
assert 'models' in files, 'models directory not found! {}'.format(files)
assert 'project_config.yaml' in files, 'project config not found! {}'.format(files)

cfg = configuration.make_feature_extractor_inference_cfg(project_path=project_path, preset=preset)
print(OmegaConf.to_yaml(cfg))

cfg.feature_extractor.weights = 'latest'
cfg.flow_generator.weights = 'latest'

cfg.inference.overwrite = True
# make sure errors are thrown
cfg.inference.ignore_error = False
cfg.compute.num_workers = 8 #EDIT HERE

print(OmegaConf.to_yaml(cfg))

feature_extractor_inference(cfg)

# this just parses our DATA directory, to get the path to each file for each video
records = projects.get_records_from_datadir(os.path.join(project_path, 'DATA'))
animal = random.choice(list(records.keys()))
record = records[animal]

# I call the file output by inference the `outputfile` in various places in the code
outputfile = record['output']

utils.print_hdf5(outputfile)

# we use the h5py package for this
with h5py.File(outputfile, 'r') as f:
  probabilities = f['resnet18/P'][:]
# n frames x K behaviors
print(probabilities.shape)
probabilities