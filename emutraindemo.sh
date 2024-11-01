#!/bin/bash
#SBATCH --job-name=cmbemu
#SBATCH --output=./output/cmbemu.txt
#SBATCH --time=48:00:00
#SBATCH -p a100-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gpus=1
#SBATCH --mem=200g


echo "Available CPUs:  $SLURM_JOB_CPUS_PER_NODE"
# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1



echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID

cd $SLURM_SUBMIT_DIR

conda activate cocoapy39
export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

python ./cmbemutrflhsTT.py\
        -p Demo/cos_omm_train.npy \
        -t Demo/cos_omm_train_TT.npy \
        -q Demo/cos_omm_vali.npy \
        -v Demo/cos_omm_vali_TT.npy \
        -e Demo/extra_demo_omm.npy  \
        -o ./trainedemu/chiTTtestc
