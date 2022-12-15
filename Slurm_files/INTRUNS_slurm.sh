#!/bin/bash
# ###### Some commands https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands/
# ###### Multiprocessing in slurm https://docs.hpc.cofc.edu/using-the-hpc/scheduling-jobs/execute-a-job
# ###### Multiprocessing in slurm https://docs.hpc.cofc.edu/using-the-hpc/scheduling-jobs/execute-a-job/python
# ###### For a runnning file https://slurm.schedmd.com/sstat.html
# ###### for a job done https://slurm.schedmd.com/sacct.html
# ###### SLURM PARAMETERS ##########
#
#SBATCH --job-name=I_IP7615                 #Nombre del job
#SBATCH -p medium                       #Cola a usar, Default=short 
#SBATCH -N 1                                #Nodos requeridos, Default=1
#SBATCH -n 1                                #Tasks paralelos, recomendado para MPI, Default=1
#SBATCH --cpus-per-task=1                   #Cores requeridos por task, recomendado para multi-thread, Default=1
#SBATCH --mem=30000                       #Memoria en Mb por CPU, Default=2048
#SBATCH --time=20:00:00                   #Tiempo máximo de corrida, Default=2 horas
#SBATCH --mail-user=b.leon@uniandes.edu.co  #Correo para enviar aviso
#SBATCH --mail-type=ALL
#SBATCH -o output_IntRunsIP7615.o1                #Nombre de archivo de salida

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
# export HDF5_USE_FILE_LOCKING='FALSE'
time python $remote_directory/PATENTES/Patents_clear/Python_files/Calculating_SPNP_Integrate_Runs.py -c IPC -start 1976 -stop 2015

echo "-----------------------------"
echo -e "Sbatch done"

echo "JOB ID:"$SLURM_JOB_ID
#####################################