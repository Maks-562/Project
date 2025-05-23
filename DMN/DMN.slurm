#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4571mb
#SBATCH --time=00:02:00


# Change into the directory where I submitted the job
cd $SLURM_SUBMIT_DIR 

module purge
module load GCC/13.2.0

module load SciPy-bundle/2023.11 sympy/1.12  matplotlib/3.8.2

export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH
 
which python
python3 --version
python3 -c "import numpy; print(numpy.__version__)"




export N=2

export lr=0.01
export lam=0

export epoch_num=5

export optimiser='ADAM'




python3 DMN_train.py $N $lr $lam $epoch_num $optimiser


exit 0
