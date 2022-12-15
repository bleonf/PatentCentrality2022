# PatentCentrality2022
Calculation and randomization of patent centralities up to year 2020

This repository contains 4 python files with which all the project is run. These four files are:

 - Reorganize_Data.py
 - Calculating_EMPIRICAL_SPNP.py
 - Calculating_SPNP_Integrate_Runs.py
 - Running_Jobfiles.py

Also, in the repository, executable files so the project can run under a HPC managed with SLURM. These executables with their functions are:
 - REORGANIZE_slurm.sh
    - Runs Reorganize_Data.py. Uses class input (-c option) to organize class data (patents and citations) into the desired format  
 - EMPIRICAL_slurm.sh
    - Runs Calculating_EMPIRICAL_SPNP.py with the default empirical setting for a class (-c option) and initial and final years (-start -stop options)  
 - INTRUNS_slurm.sh
    - Runs Calculating_SPNP_Integrate_Runs.py. Integrates randomizations for a class (-c option) and initial and final years (-start -stop options)
 - RunJobs_slurm.sh
    - Runs Running_Jobfiles.py. Using python multiprocessing runs in parallel n-randomizations (option -n) for a class (-c option) and initial and final years (-start -stop options). Also runs  Calculating_SPNP_Integrate_Runs.py to integrate randomizations sequentially, the same as INTRUNS_slurm.sh.

The original code where this implementation was based upon can be found at https://github.com/jeffalstott/patent_centralities. Repository by Jeff Alstott.

PyBiRewire module for download and installation is written by Andrea Gobbi and can be found at https://github.com/andreagobbi/pyBiRewire

A youtube playlist on how to use the cluster at Universidad de los Andes to run this project can be found at: https://youtube.com/playlist?list=PLPad7rgmV3bRevOhTNh6zsUlupPgnvnBj
