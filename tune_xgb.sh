#!/bin/bash
#SBATCH --job-name=als_xgb
#SBATCH --partition=AMD
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=xgb_%j.log

cd ~/projects/ALS_Bachelorthesis
source .venv/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -m src.models.tune --model xgb --n-jobs 4
