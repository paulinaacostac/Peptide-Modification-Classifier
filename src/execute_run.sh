#!/bin/bash
#SBATCH --job-name="modifications classifier"
#SBATCH --output="labeler-out.%j.%N.out"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=256
#SBATCH --mem=2000gb
#SBATCH --account=fsaeed
#SBATCH --no-requeue
#SBATCH -t 10:00:00

python sbatch_run_train.py
