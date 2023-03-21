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

print("Imported deepethogram feature extractor")

project_path = '/n/data1/hms/neurobio/sabatini/Janet/Kim_segmented_deepethogram' #EDIT HERE
files = os.listdir(project_path)
assert 'DATA' in files, 'DATA directory not found! {}'.format(files)
assert 'models' in files, 'models directory not found! {}'.format(files)
assert 'project_config.yaml' in files, 'project config not found! {}'.format(files)

preset = 'deg_f'
cfg = configuration.make_feature_extractor_train_cfg(project_path, preset=preset)
print(OmegaConf.to_yaml(cfg))

cfg.flow_generator.weights = 'latest'
cfg.compute.num_workers = 8 #n_cpus  EDIT HERE

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

feature_extractor = feature_extractor_train(cfg)

model_path = os.path.join(project_path, 'models')
weights = projects.get_weights_from_model_path(model_path)
flow_weights = weights['feature_extractor']
# because we used deg_f, our model type is a resnet18
latest_weights = flow_weights['resnet18'][-1]
# our run directory is two steps above the weight file
run_dir = os.path.dirname(os.path.dirname(latest_weights))
assert os.path.isdir(run_dir), 'run directory not found! {}'.format(run_directory)

figure_dir = os.path.join(run_dir, 'figures')
figure_files = utils.get_subfiles(figure_dir, 'file')
assert len(figure_files) >= 1

#Image(figure_files[0])
print("Finished training feature extractor")