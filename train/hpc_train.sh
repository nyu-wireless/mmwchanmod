#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=5:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=gen_linear
#SBATCH --mail-type=END
#SBATCH --mail-user=123@abc.xyz
#SBATCH --output=slurm_%A_%a.out

module load python3/intel/3.7.3
python3 train_mod.py --nepochs_path 2000  --model_dir model_data
