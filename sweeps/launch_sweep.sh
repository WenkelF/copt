#!/usr/bin/env bash

#SBATCH --array=1-48%12

## Name of your SLURM job
# SBATCH --job-name=run

## Files for logs: here we redirect stoout and sterr to the same file
#SBATCH --output=outputs/out_%x_%j_%a.out   # %x=job-name, %j=jobid, %a=array-id
#SBATCH --error=outputs/error_%x_%j_%a.out
#SBATCH --open-mode=append

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

## Time limit for the job
#SBATCH --time=12:00:00

#SBATCH --mem=40Gb

## Number of GPUs to use
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:48gb:1

## Partition to use,
#SBATCH --partition=long
  
set -e

cd /home/mila/f/frederik.wenkel/projects/co_dev

module load miniconda/3 cuda/11.8

source /home/mila/f/frederik.wenkel/.bashrc

conda activate /home/mila/f/frederik.wenkel/miniconda3/envs/copt

wandb agent --count 1 wenkelf/co_expts/1bqn6vjv