#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4571mb
#SBATCH --time=20:00:00
#SBATCH --array=0-49


# Change into the directory where I submitted the job
cd $SLURM_SUBMIT_DIR 

module purge
module load GCC/13.2.0

module load SciPy-bundle/2023.11 sympy/1.12 

export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH




python3 -c "import numpy; print(numpy.__version__)"




export N=5

# export lr=10E-3
# export lam=10E-3
# lr=$(awk "NR==$((SLURM_ARRAY_TASK_ID+1)) {print \$1}" params.txt)
# lam=$(awk "NR==$((SLURM_ARRAY_TASK_ID+1)) {print \$2}" params.txt)
# read lr lam < <(awk "NR==$((SLURM_ARRAY_TASK_ID+1))" params.txt)

# TASK_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" params.txt)

# lr=$(echo $TASK_LINE | awk '{print $1}')
# lam=$(echo $TASK_LINE | awk '{print $2}')

line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" params.txt)

lr=$(echo "$line" | cut -d',' -f1)
lam=$(echo "$line" | cut -d',' -f2)

echo "Task ID = $SLURM_ARRAY_TASK_ID"
echo "lr = $lr"
echo "lam = $lam"

export epoch_num=10000

export optimiser='SGD'




# python3 DMN_train.py $N $lr $lam $epoch_num $optimiser
python3 DMN_train.py $N $lr $lam $epoch_num $optimiser > output_task_${SLURM_ARRAY_TASK_ID}.log 2>&1



export optimiser='ADAM'
# python3 DMN_train.py $N $lr $lam $epoch_num $optimiser
python3 DMN_train.py $N $lr $lam $epoch_num $optimiser > output_task_${SLURM_ARRAY_TASK_ID}.log 2>&1

exit 0
