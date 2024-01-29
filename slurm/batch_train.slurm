#!/bin/bash

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=12        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:p100:2
#SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH --mail-user=clee@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

### Change here
#SBATCH -J "resnets"   # job name

source /home/clee/miniconda3/etc/profile.d/conda.sh
conda activate moco

### Change here
bash bash/train.sh resnet18 tcp://localhost:10003 /central/scratch/clee/thermal_ssl
bash bash/train.sh resnet50 tcp://localhost:10005 /central/scratch/clee/thermal_ssl