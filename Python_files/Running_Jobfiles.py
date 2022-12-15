"""
Runs the 1000 randomization_ID_USPC.py files located in src/jobfiles/
Set missing list if you are running files that were not run in the original pass (memory error or parsing error)
"""
# from cProfile import run
import os
import glob
from multiprocessing import Pool, cpu_count,Process
import time
import re
import argparse

parser = argparse.ArgumentParser(description='A test program.')

parser.add_argument("-c", "--Patent_Class", help="Enter USPC,CPC or IPC", type=str)
parser.add_argument("-start", "--First_Year", help="Enter first year to filter Year", type=int)
parser.add_argument("-stop", "--Last_Year", help="Enter last year to filter Year", type=int)
parser.add_argument("-f", "--first_rand", help="Enter last rand id", type=int)
parser.add_argument("-l", "--last_rand", help="Enter first rand id", type=int)
parser.add_argument("-n", "--Core_Number", help="Enter number of cores", type=int)
args = parser.parse_args()

#Defaults:
n=10
if args.Core_Number:
    n=args.Core_Number

FIRST_YEAR=1976 #Default
if args.First_Year:
    FIRST_YEAR=args.First_Year
    
LAST_YEAR=2015 #Default
if args.Last_Year:
    LAST_YEAR=args.Last_Year
    
classes=["USPC","IPC","CPC"]
class_system=classes[1] #Default
if args.Patent_Class:
    class_system=args.Patent_Class
    
first_rand=1
last_rand=1000
if args.first_rand:
    first_rand=args.first_rand
if args.last_rand:
    lastt_rand=args.last_rand

# path=f'/hpcfs/home/educacion/b.leon/PATENTES/Patents_clear/Input_data/'+class_system+'/Controls/jobfiles/jobs'+First_year+Last_year+'/r*'
out_path=f'/hpcfs/home/educacion/b.leon/PATENTES/Patents_clear/Input_data/'+class_system+'/Controls/Synthetic_controls/{FIRST_YEAR}{LAST_YEAR}/*.csv'
print(f"CPU Count: {cpu_count()}") 
print("Setup ready for jobfiles")

done_list=glob.glob(out_path) #List with all done files 
done_rands=[]
for done_path in done_list:
    try:
        done_rands.append(int(done_path[:-4].split("_")[-1]))
    except ValueError:
        print(f"{done_path} does not end with its randomization_id!")
        continue
Missing_elements=[i for i in range(first_rand,last_rand+1) if i not in done_rands] #randomization ids not in the out_path directory
    

def RunFromEmpirical(id):
    time_file=time.time()    
    print("Running Function")
    os.system(f"python /hpcfs/home/educacion/b.leon/PATENTES/Patents_clear/Python_files/Calculating_EMPIRICAL_SPNP.py -id {id} -r -start {FIRST_YEAR} -stop {LAST_YEAR} -c {class_system}")
    time_end=time.time()
    return f'Randomization_{id} Done! after {time_end-time_file:.3f} seconds'

#Simple multiprocessing tutorial at 
# https://doku.lrz.de/display/PUBLIC/FAQ%3A+Embarassingly+parallel+jobs+using+Python+and+R
# https://www.youtube.com/watch?v=fKl2JW_qrso&ab_channel=CoreySchafer

pool=Pool(n)
results=pool.map(RunFromEmpirical,Missing_elements)
pool.close()
pool.join()
for result in results:
    print(result)
print("Processes done")


# def RunJobFile(file):
#     time_file=time.time()    
#     print(f"OMG! {file} is working, starting after {time_file-time_start} \n")
#     os.system("python "+file)
#     time_end=time.time()
#     print(f"OMG! {file} ran after {time_end-time_file} seconds\n ")
#     return file+" Done!"

# JobFiles_list=['/hpcfs/home/educacion/b.leon/PATENTES/Patents_clear/Input_data/'+class_system+'/Controls/jobfiles/jobs'+First_year+Last_year+'/randomization_'+str(i)+f'_{class_system}.py' for i in Missing_elements]     
# print(JobFiles_list)
# print(len(JobFiles_list))
# print("Joblist done!")

# processes={}

# for i in range(len(JobFiles_list)):
#     processes[i]=Process(target=RunJobFile,args=[JobFiles_list[i]])

# print("Processes starting")
# for i in range(len(JobFiles_list)):
#     processes[i].start()  

# print("Processes joining")
# for i in range(len(JobFiles_list)):
#     processes[i].join()



