#!/usr/bin/env bash

#SBATCH --array=1-1

## Name of your SLURM job
# SBATCH --job-name=run

## Files for logs: here we redirect stoout and sterr to the same file
#SBATCH --output=outputs/out_%x_%j_%a.out   # %x=job-name, %j=jobid, %a=array-id
#SBATCH --error=outputs/error_%x_%j_%a.out
#SBATCH --open-mode=append

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

## Time limit for the job
#SBATCH --time=10:00:00

#SBATCH --mem=50Gb

## Partition to use,
#SBATCH --partition=long-cpu

set -e

cd /home/mila/f/frederik.wenkel/projects/copt_graphgym

module load miniconda/3 cuda/11.8

source /home/mila/f/frederik.wenkel/.bashrc

conda activate /home/mila/f/frederik.wenkel/miniconda3/envs/copt

wandb agent --count 1 wenkelf/copt_graphgym/v3dym4tr