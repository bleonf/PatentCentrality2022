
# coding: utf-8
"""
runs with INT_RUNS_slurm.sh
inputs	empirical.h5
outputs h5 files in summary statistics after creating and running many files
    empirical
    randomized_mean
    randomized_std
    empirical_percentile
    empirical_z_scores

"""



import gc
import time 
import pandas as pd
from pylab import *
from numpy import sqrt
from numpy import log as log1
import glob
import argparse

parser = argparse.ArgumentParser(description='Program that does the empirical SPNP calculation or a randomized control by command line arguments')

parser.add_argument("-c", "--Patent_Class", help="Enter USPC,CPC or IPC", type=str)
parser.add_argument("-start", "--First_Year", help="Enter first year to filter Year", type=int)
parser.add_argument("-stop", "--Last_Year", help="Enter last year to filter Year", type=int)
args = parser.parse_args()
pd.options.display.max_columns = 50
pd.options.display.precision = 3


tt=time.time()
data_directory = '/hpcfs/home/educacion/b.leon/PATENTES/Patents_clear/Input_data/'
classes=["USPC","IPC","CPC"]
Update=False#In testing to update randomization
class_system=classes[0]
if args.Patent_Class:
    class_system=args.Patent_Class
controls_directory = data_directory+class_system+'/Controls/Synthetic_controls/'

FIRST_YEAR=1976
if args.First_Year:
    FIRST_YEAR=args.First_Year
LAST_YEAR=2015

if args.Last_Year:
    LAST_YEAR=args.Last_Year
# ,patent_number,SPNP_count_2015,outgoing_path_count_log_2015,incoming_path_count_log,filing_year,filing_year+2,SPNP_count_2,incoming_path_count_log_2,filing_year+3,SPNP_count_3,incoming_path_count_log_3,filing_year+5,SPNP_count_5,incoming_path_count_log_5,filing_year+8,SPNP_count_8,incoming_path_count_log_8,meanSPNPcited,stdSPNPcited,meanSPNPcited_1year_before,stdSPNPcited_1year_before,Citing_Patent,mean_incoming_path_count_log_cited,std_incoming_path_count_log_cited




empirical = pd.read_csv(data_directory+class_system+'/'+class_system+f'_empirical_{FIRST_YEAR}_{LAST_YEAR}.csv',low_memory=False).set_index("patent_number").drop(columns=["Citing_Patent", "Unnamed: 0"]).astype(float)
#,patent_number,SPNP_count_2015,outgoing_path_count_log_2015,incoming_path_count_log,filing_year,filing_year+2,SPNP_count_2,incoming_path_count_log_2,filing_year+3,SPNP_count_3,incoming_path_count_log_3,filing_year+5,SPNP_count_5,incoming_path_count_log_5,filing_year+8,SPNP_count_8,incoming_path_count_log_8,meanSPNPcited,stdSPNPcited,meanSPNPcited_1year_before,stdSPNPcited_1year_before,Citing_Patent,mean_incoming_path_count_log_cited,std_incoming_path_count_log_cited
#,patent_number,SPNP_count_2015,outgoing_path_count_log_2015,incoming_path_count_log,filing_year,filing_year+2,SPNP_count_2,incoming_path_count_log_2,filing_year+3,SPNP_count_3,incoming_path_count_log_3,filing_year+5,SPNP_count_5,incoming_path_count_log_5,filing_year+8,SPNP_count_8,incoming_path_count_log_8,meanSPNPcited,stdSPNPcited,meanSPNPcited_1year_before,stdSPNPcited_1year_before,Citing_Patent,mean_incoming_path_count_log_cited,std_incoming_path_count_log_cited


print("Empirical data:")
print(empirical.shape)
print(empirical.info())

Controls_list=glob.glob(controls_directory+f'{FIRST_YEAR}{LAST_YEAR}/synthetic_control_*.csv')
print("Length of controls list:")
print(len(Controls_list))
def running_stats(empirical,Path_list=Controls_list):

    Mean = None
    Var=0
    k=0.0
    p=0
    all_max = None
    all_min = None
    for Cont in Path_list:
        try:
            x = pd.read_csv(Cont,low_memory=False).set_index("patent_number").drop(columns=["Citing_Patent", "Unnamed: 0"]).astype(float)
            x.columns=empirical.columns
            # x.SPNP_count_2015=x.SPNP_count_2015.apply(log1)
            if k==0:
                print("First randomization data:")
                print(x.shape)
                print(x.info())
            print(f'shape:{x.shape}')
        except Exception:
            print("Data not loading for %s. Continuing."%Cont)
            continue
            
        if Mean is None:
            Mean = x
            Var = 0
            p = 0
            k = 0
            continue
        k += 1.0
        M_previous = Mean

        Mean = M_previous+((x.subtract(M_previous))/k)
        Var += (x.subtract(M_previous))*(x.subtract(Mean))
        p += (empirical>x).astype('int')
        gc.collect()  
    standard_deviation = sqrt(Var/(k-1))
    print("Dtypes at error")
    print(type(standard_deviation))
    print(type(Var))
    print(type(k))
    z_scores = (empirical.subtract(Mean))/standard_deviation
    t = time.time()
    print(k)
    print(f'total time: {(t-tt)/60} minutes')
    return Mean, Var, p/k, z_scores

    
M, standard_deviation, empirical_percentile, empirical_z_scores = running_stats(empirical, Controls_list)


print("Done running stats! this took {} seconds".format(time.time()-tt))
for df in [M, standard_deviation, empirical_percentile, empirical_z_scores]:
    df['patent_number'] = empirical.reset_index()['patent_number']
    df['filing_year'] = empirical['filing_year']
    
print(M.describe())
print(standard_deviation.describe())
print(empirical_percentile.describe())
print(empirical_z_scores.describe())


print("Done customizing Dataframes! this took {} seconds".format(time.time()-tt))
z_scores = empirical_z_scores
z_scores.values[where(z_scores==inf)]=nan 
z_scores.values[where(z_scores==-inf)]=nan 

standard_deviation.to_csv(data_directory+class_system+'/'+f'{class_system}_{FIRST_YEAR}{LAST_YEAR}_std.csv')
M.to_csv(data_directory+class_system+'/'+f'{class_system}_{FIRST_YEAR}{LAST_YEAR}_mean.csv')
empirical_percentile.to_csv(data_directory+class_system+'/'+f'{class_system}_{FIRST_YEAR}{LAST_YEAR}_perc.csv')
empirical_z_scores.to_csv(data_directory+class_system+'/'+f'{class_system}_{FIRST_YEAR}{LAST_YEAR}_zs.csv')
# store = pd.HDFStore(data_directory+class_system+'/'+f'{class_system}_{FIRST_YEAR}{LAST_YEAR}_summary_statistics.h5', mode='a', table=True)

# store.put('/randomized_mean_%s'%(class_system), M, 'table', append=False)
# store.put('/randomized_std_%s'%(class_system), standard_deviation, 'table', append=False)
# store.put('/empirical_percentile_%s'%(class_system), empirical_percentile, 'table', append=False)
# store.put('/empirical_z_scores_%s'%(class_system), z_scores, 'table', append=False)
# store.put('/empirical', empirical, 'table', append=False)

# # store.put('/randomized_min_%s'%(class_system), all_min, 'table', append=False)
# store.close()


#All the cases where the z-scores are inf is where the 1,000 randomized controls said there should be 0 deviation, BUT
#the empirical case was different anyway. In each of these cases, the empirical case was JUST slightly off. Sometimes
#a floating point error, and sometimes off by 1 (the minimal amount for citation counts). We shall treat this as not actually
#deviating, and so it becomes 0/0, which is equal to nan.

#Change to number of iterations done
initialK=0

def Update_stats(empirical,new,currentM,currentS,currentPerc,k):
    new.columns=empirical.columns
    M = currentM + ((new.subtract(currentM))/k)
    S = currentS + (new.subtract(currentM))*(new.subtract(M))
    p=currentPerc+(empirical>new).astype('int')
    gc.collect()
    standard_deviation = sqrt(S/(k-1))
    z_scores = (empirical.subtract(M))/standard_deviation
    return M, S, p/k, z_scores,k+1

if Update:  
    
    currentM=pd.read_csv(data_directory+class_system+'/'+f'{class_system}_{FIRST_YEAR}{LAST_YEAR}_mean.csv')
    currentS=pd.read_csv(data_directory+class_system+'/'+f'{class_system}_{FIRST_YEAR}{LAST_YEAR}_std.csv')
    currentPerc=pd.read_csv(data_directory+class_system+'/'+f'{class_system}_{FIRST_YEAR}{LAST_YEAR}_perc.csv')
    
    for Cont in Controls_list:
        New=pd.read_csv(Cont).set_index("patent_number").drop(columns=["Citing_Patent", "Unnamed: 0"]).astype(float)
        currentM,currentS,currentPerc,currentZ,initialK=Update_stats(empirical,New,currentM,currentS,currentPerc,initialK)

    currentS.to_csv(data_directory+class_system+'/'+f'{class_system}_{FIRST_YEAR}{LAST_YEAR}_std.csv')
    currentM.to_csv(data_directory+class_system+'/'+f'{class_system}_{FIRST_YEAR}{LAST_YEAR}_mean.csv')
    currentPerc.to_csv(data_directory+class_system+'/'+f'{class_system}_{FIRST_YEAR}{LAST_YEAR}_perc.csv')
    currentZ.to_csv(data_directory+class_system+'/'+f'{class_system}_{FIRST_YEAR}{LAST_YEAR}_zs.csv')