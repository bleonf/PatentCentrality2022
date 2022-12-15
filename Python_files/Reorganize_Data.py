"""Python script that joins PATENT_INFO_1926_2020.csv with the USPC, IPC or CPC classification and with PATENT_CITATION_1926_2020_NoNeg_CitLag.csv

Raises:
    ValueError: If error when merging or floatifying mainclass ID
Creates:
    citations.csv/citations.h5 with all citations with Citing patents, cited patents, classes and filing years
    patents.csv/patents.h5 filtered patents that are included in citations
"""
# coding: utf-8

import pandas as pd
import argparse
from numpy import nan
parser = argparse.ArgumentParser(description='Program that does the empirical SPNP calculation or a randomized control by command line arguments')
parser.add_argument("-c", "--Patent_Class", help="Enter USPC,CPC or IPC", type=str)
args = parser.parse_args()


data_input='/hpcfs/home/educacion/b.leon/PATENTES/Patents_clear/Input_data/'
data_output='/hpcfs/home/educacion/b.leon/PATENTES/Patents_clear/Input_data/'

classes=["USPC","IPC","CPC"]
classification=classes[0]
if args.Patent_Class:
    classification=args.Patent_Class
print("-----------------------------------------------------------")    
print(classification)
print("-----------------------------------------------------------")    

patents = pd.read_csv(data_input+'PATENT_INFO_1926_2020.csv', 
                 usecols=['patent_number', 'filing_year'],low_memory=False)
# patents=patents[patents['filing_year']>=1950]
print("Patents dataframe:")
print(patents.head())
print("Patents description")
print(patents.describe())

if classification=="IPC":    
    IPC_INFO = pd.read_csv(data_input+"ipcr.tsv.zip",
                                sep='\t',usecols=['patent_id','section','ipc_class','subclass','main_group','subgroup','sequence'],low_memory=False, compression='zip')
    print("IPC_INFO head:")
    print(IPC_INFO.head())
    IPC_INFO = IPC_INFO[IPC_INFO['sequence']==0]
    IPC_INFO['IPC4'] = IPC_INFO['section'].astype(str)+IPC_INFO['ipc_class'].astype(str)+IPC_INFO['subclass'].astype(str)
    IPC_INFO['IPC6'] = IPC_INFO['IPC4'].astype(str)+IPC_INFO['main_group'].astype(str)

    patents=pd.merge(patents,IPC_INFO[["patent_id","IPC4","IPC6","sequence"]],
                        left_on="patent_number",right_on="patent_id")
    # patents=patents.drop(columns=["IPC6","sequence"])
    patents=patents.rename(columns={"IPC4":"mainclass_id"})
    
elif classification=="USPC": #Done
    USPC_INFO=pd.read_csv(data_input+"uspc.tsv",
                        sep='\t',usecols=['patent_id','mainclass_id','subclass_id','sequence'],low_memory=False)
    patents=pd.merge(patents,USPC_INFO[['patent_id','mainclass_id','subclass_id','sequence']],
                        left_on="patent_number",right_on="patent_id")
    
elif classification=="CPC": #Done
    CPC_INFO=pd.read_csv(data_input+"cpc_current.tsv.zip",
                        sep='\t',low_memory=False, compression='zip',
                        usecols=['patent_id','section_id','subsection_id','sequence']
                        )
    print(CPC_INFO.sample(10))
    print(CPC_INFO.info())
    # exit()
    patents=pd.merge(patents,CPC_INFO[['patent_id','section_id','subsection_id','sequence']].astype({"patent_id":"str"}),
                        left_on="patent_number",right_on="patent_id")
else:
    print("The classification system is not recognized! Default exit!" +classification)
    exit()
    


print("\n""Patents dataframe after merge with classification:")
print("Columns:")
print(patents.columns)
print("Head:")
print(patents.head())
print("Patents info")
print(patents.info())

print("\n""Columns after rename:")
patents.columns=['patent_number', 'filing_year', 'patent_id', 'Class','subclass_id', 'sequence']
print(patents.columns)

def floatify(x):
    try:
        return float(x)
    except ValueError:
        return nan
patents['Class']=patents['Class'].apply(floatify)
# vc = patents['Class'].value_counts()
# from pylab import array
# vc[array(list(map(type, vc.index.values)))==str].sum()

# patents['Class'] = patents['mainclass_id'].apply(floatify)


citations = pd.read_csv(data_input+'CITATION_INFO_1950_2020_noNegCitlag.csv', 
                usecols=['citing', 'cited','filing_year_cited', 'filing_year_citing'])

citations.rename(columns={  "citing": 'Citing_Patent',
                            "cited": 'Cited_Patent',
                            "filing_year_citing": 'Year_Citing_Patent',
                            "filing_year_cited": 'Year_Cited_Patent',
                             }, inplace=True)


### Drop citations that are erroneous, having a citation that points to a future patent
citations = citations[citations['Year_Citing_Patent']>=citations['Year_Cited_Patent']]
citations["Cited_Patent"]=citations["Cited_Patent"].astype(str)
citations["Citing_Patent"]=citations["Citing_Patent"].astype(str)
patents["patent_number"]=patents["patent_number"].astype(str)

print("\n""114: Citations")
print(citations.info())
print(citations.describe())
print(citations.isna().sum())
print("\n""117: patents")
print(patents.info())
print(patents.describe())
print(patents.isna().sum())



#Filter patents that are not included in citations dataframe
patents = patents[patents['patent_number'].isin(citations['Cited_Patent']) | patents['patent_number'].isin(citations['Citing_Patent'])]
patents=patents.drop_duplicates(subset='patent_number')
# citations=citations[citations['Citing_Patent'].isin(patents['patent_number']) & citations['Cited_Patent'].isin(patents['patent_number']) ]

if patents.shape[0]<10:
    print("Patents:")
    print(patents)
    raise ValueError("Patents results in a datframe with {} rows".format(patents.shape[0]))
print("Patents results in a dataframe with {} rows".format(patents.shape[0]))


# #Class assign by chunks of 2_000_000 patents
# for patent_type in ['Citing_Patent', 'Cited_Patent']:
#     print(patent_type)
#     z = empty(citations.shape[0])
#     start_ind = 0
#     while start_ind<z.shape[0]:
#         print(start_ind)
#         stop_ind = start_ind+2000000
#         z[start_ind:stop_ind] = patents.loc[citations[start_ind:stop_ind][patent_type], 'Class']
#         start_ind = stop_ind+1
#     citations['Class_%s'%patent_type] = z

# citations.dropna(inplace=True)

######################################
# Creating final citations dataframe #
######################################
citations=pd.merge(citations,patents[["patent_number","Class"]],how="left",left_on="Citing_Patent",right_on="patent_number")
citations=pd.merge(citations,patents[["patent_number","Class"]],how="left",left_on="Cited_Patent",right_on="patent_number",suffixes=["_citing","_cited"])
#Here, citations with patents with no class are still preserved, bt with empty class fields
citations=citations.drop(columns=["patent_number_citing","patent_number_cited"]).drop_duplicates(subset=["Cited_Patent","Citing_Patent"])
citations.rename({"Class_citing":"Class_Citing_Patent","Class_cited":"Class_Cited_Patent"})

##############################
# CItations dataframe export #
##############################
print("--------------------------------------")
# citations.to_hdf(data_output+classification+'/citations.h5', 'df', complevel=9, complib='blosc')
citations.to_csv(data_output+classification+'/'+classification+'_citations.csv')
print("Citations to csv done with the columns and types:")
print(citations.describe())
print(citations.dtypes)
print("--------------------------------------")

############################
# Patents dataframe export #
############################
print("--------------------------------------")
patents.drop_duplicates(subset=["patent_number"],inplace=True)
# patents.to_hdf(data_output+classification+'/patents.h5', 'df', complevel=9, complib='blosc')
patents.to_csv(data_output+classification+'/'+classification+'_patents.csv')
print("Patents to csv done with the columns and dtypes:")
print(patents.describe())
print(patents.dtypes)
print("--------------------------------------")

# real    59m2.221s
# user    53m37.235s
# sys     4m38.312s


