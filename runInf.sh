#!/bin/bash
#SBATCH -c 8                             # number of cpus
#SBATCH -t 0-08:00                      # Runtime in D-HH:MM format
#SBATCH -p gpu                          # Partition to run in
#SBATCH --gres=gpu:1                    # number of gpus
#SBATCH --mem=64G                      # Memory total in MiB (for all cores)
#SBATCH -o %j_runInf.out               # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e %j_runInf.err               # File to which STDERR will be written, including job ID (%j)
#SBATCH --mail-type=END


module load gcc/6.2.0
module load python/3.7.4
unset PYTHONPATH
source deepethogram/bin/activate

python3 ~/deepethogram/de_runInference.py