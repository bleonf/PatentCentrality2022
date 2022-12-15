#!/bin/bash
# ###### Some commands https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands/
# ###### Multiprocessing in slurm https://docs.hpc.cofc.edu/using-the-hpc/scheduling-jobs/execute-a-job
# ###### Multiprocessing in slurm https://docs.hpc.cofc.edu/using-the-hpc/scheduling-jobs/execute-a-job/python
# ###### For a runnning file https://slurm.schedmd.com/sstat.html
# ###### for a job done https://slurm.schedmd.com/sacct.html
# ###### SLURM PARAMETERS ##########

#SBATCH --job-name=REORGALL                     #Nombre del job
#SBATCH -p long                    #Cola a usar, Default=short 
#SBATCH -N 1                                #Nodos requeridos, Default=1
#SBATCH -n 3                                #Tasks paralelos, recomendado para MPI, Default=1
#SBATCH --cpus-per-task=1                   #Cores requeridos por task, recomendado para multi-thread, Default=1
#SBATCH --mem=45000                    #Memoria en Mb por CPU, Default=2048
#SBATCH --time=06:00:00                     #Tiempo máximo de corrida, Default=2 horas
#SBATCH --mail-user=b.leon@uniandes.edu.co  #Correo para enviar aviso
#SBATCH --mail-type=ALL
#SBATCH -o output_REORGANIZEALL.o1                #Nombre de archivo de salida
#SBATCH -e error_REORGANIZEALL.e1

#####################################

# ###### CODE EXECUTION #############

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
time python $remote_directory/PATENTES/Patents_clear/Python_files/Reorganize_Data.py -c USPC
time python $remote_directory/PATENTES/Patents_clear/Python_files/Reorganize_Data.py -c IPC
time python $remote_directory/PATENTES/Patents_clear/Python_files/Reorganize_Data.py -c CPC


echo "-----------------------------"
echo -e "Sbatch done"
echo "JOB ID:"$SLURM_JOB_ID
#####################################
# real    59m2.221s
# user    53m37.235s
# sys     4m38.312s
