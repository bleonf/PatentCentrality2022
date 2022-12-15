#!/bin/bash
# ###### Some commands https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands/
# ###### Multiprocessing in slurm https://docs.hpc.cofc.edu/using-the-hpc/scheduling-jobs/execute-a-job
# ###### Multiprocessing in slurm https://docs.hpc.cofc.edu/using-the-hpc/scheduling-jobs/execute-a-job/python
# ###### For a runnning file https://slurm.schedmd.com/sstat.html
# ###### for a job done https://slurm.schedmd.com/sacct.html
# ###### SLURM PARAMETERS ##########
#SBATCH --job-name=E_CP7615                  #Nombre del job
#SBATCH -p medium                           #Cola a usar, Default=short 
#SBATCH -N 1                                #Nodos requeridos, Default=1
#SBATCH -n 1                                #Tasks paralelos, recomendado para MPI, Default=1
#SBATCH --cpus-per-task=1                   #Cores requeridos por task, recomendado para multi-thread, Default=1
#SBATCH --mem=55000                         #Memoria en Mb por CPU, Default=2048
#SBATCH --time=06:59:00                     #Tiempo máximo de corrida, Default=2 horas
#SBATCH --mail-user=b.leon@uniandes.edu.co  #Correo para enviar aviso
#SBATCH --mail-type=ALL
#SBATCH -o output_E_CPC7615.o1                #Nombre de archivo de salida

#####################################

# ###### MODULE LOAD ################


host=`/bin/hostname`
date=`/bin/date`
remote_directory="/hpcfs/home/educacion/b.leon"
echo "-----------------------------"
echo "Resource usage date: "$date
echo "Running on: "$host
echo "Running sbatch from: " 
pwd
echo "-----------------------------"
echo "Running code"
source $remote_directory/PATENTES/Patents_clear/virtualenvs/Patents_env3.9/bin/activate
export HDF5_USE_FILE_LOCKING='FALSE'
time python $remote_directory/PATENTES/Patents_clear/Python_files/Calculating_EMPIRICAL_SPNP.py -c USPC -start 1976 -stop 2015
echo "-----------------------------"
echo -e "Sbatch done"
echo "JOB ID:"$SLURM_JOB_ID
#####################################

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++ Partition   ++ Nodes             ++ MaxNodesPJ  ++ DefMemCPU  ++ MaxMemCPU  ++ MaxTime   ++ DefTime   ++ MaxCPUPU  ++ MaxMemPU   ++ MaxCPUPJ  ++ MaxGPUPU
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++ bigmem      ++ a-[4-6],i-[1-5]   ++ 2           ++ 4096       ++ 32768      ++ 15 días   ++ 2 horas   ++ 96        ++ 578845     ++ 96
# ++ short       ++ all               ++ 16          ++ 2048       ++ 4096       ++ 2 días    ++ 2 horas   ++ 240       ++ 257275     ++ 240
# ++ medium      ++ all               ++ 8           ++ 2048       ++ 22528      ++ 7 días    ++ 2 horas   ++ 192       ++ 368640     ++ 192
# ++ long        ++ a-[1-3],i-[6-10   ++ 2           ++ 2048       ++ 16384      ++ 30 días   ++ 2 horas   ++ 96        ++ 262144     ++ 96
# ++ gpu         ++ i-gpu[1,2]        ++ 2           ++ 4096       ++ 16384      ++ 15 días   ++ 2 horas   ++ 32        ++ 191776     ++ 32        ++ 3
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
