#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=10G
#SBATCH --output="log.out"
#SBATCH --job-name="blackjax_nss_test"

now=$(date)
echo "$now"

# Loading modules
# module load 2024
# module load Python/3.10.4-GCCcore-11.3.0
source /home/twouters2/projects/05_ninjax_injections/.venv/bin/activate

# Display GPU name
nvidia-smi --query-gpu=name --format=csv,noheader

# Run the script
ninjax_analysis .

echo "DONE"
