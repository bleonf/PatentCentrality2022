
# coding: utf-8
""" 
runs with CALCULATE_slurm.sh 
Change home directory rout
from command line: run this file with options -r=["rand" or""] -id=[0-1000] -c=["CPC","USPC","IPC"] -start=[YEAR] -stop=[YEAR]
inputs	CLASS_citations.csv					    CLASS_patents.csv 	        				
outputs	if randomized control==False: 			CLASS_empirical.csv
        if randomized control==True:			synthetic_control_randomization_id.csv
Last changes: 
 - Completeness filter as function to be used accordingly (line 248) used in line 285
 - Should we filter data outside the intended years (line 258)
 - On patents imports do not replace empty space with nan
 - line 274 citations value counts to count citations made by patent
"""
home_directory="/hpcfs/home/educacion/b.leon"
data_input=home_directory+'/PATENTES/Patents_clear/Input_data/'
data_output=home_directory+'/PATENTES/Patents_clear/Input_data/'
import pandas as pd
from pylab import *
import gc
import igraph as igraph
import time
from numpy import log as log1
import sys
sys.path.insert(0,f'{home_directory}/PATENTES/Patents_clear/virtualenvs/pyBiRewire-master')
import argparse

parser = argparse.ArgumentParser(description='Program that does the empirical SPNP calculation or a randomized control by command line arguments')

parser.add_argument("-c", "--Patent_Class", help="Enter USPC,CPC or IPC", type=str)
parser.add_argument("-start", "--First_Year", help="Enter first year to filter Year", type=int)
parser.add_argument("-stop", "--Last_Year", help="Enter last year to filter Year", type=int)
parser.add_argument("-r", "--Is_randomization", help="Set to 'rand' the script will create a synthetic control", action="store_true")
parser.add_argument("-T", "--IsTest", help="In case of test set this option to exit python", action="store_true")
parser.add_argument("-id", "--randomization_id", help="Takes an int to be the new randomization id", type=int)
args = parser.parse_args()

pd.set_option('display.max_columns', 100)

#Defaults:
FIRST_YEAR=1976 
if args.First_Year:
    FIRST_YEAR=args.First_Year
    
LAST_YEAR=2015
if args.Last_Year:
    LAST_YEAR=args.Last_Year
    

classes=["USPC","IPC","CPC"]
class_system=classes[1]
if args.Patent_Class:
    class_system=args.Patent_Class
    
randomization_id = "Empirical"
if args.randomization_id:
    randomization_id=args.randomization_id
    
randomized_control = False
if args.Is_randomization:
    import BiRewire
    randomized_control=True
    with open(data_output+class_system+f'/Controls/Synthetic_controls/{FIRST_YEAR}{LAST_YEAR}/Ready.txt',"a") as MyFile:
        MyFile.write(f"\n {randomization_id}: Module imports done")    

if args.IsTest:
    print("Completed test succesfully!")
    exit()
    
start_time = time.time()
#####################
# Defining function #
#####################

print("Defining functions")

def create_patent_citation_graph(PATENT_INFO, citations):
    """Creates a patent-citation graph G where nodes are the patents and the edges are citations

    Args:
        PATENT_INFO (pd.DataFrame): Dataframe containing filing year, class, patent id and patent number
        citations (pd.DataFrame): Dataframe containing citing and cited patents' id, filing year and class

    Returns:
        igraph graph: Graph with patents as nodes and citations as connecting edges
    """    
    print('create igraph object and populate it with patent attributes')
    G = igraph.Graph(directed=True)

    ##### add nodes ####
    print('step 1: adding nodes')
    num_nodes = shape(PATENT_INFO)[0]
    # node_codes=PATENT_INFO["patent_number"].to_list()
    G.add_vertices(num_nodes)
    # G.add_vertices(node_codes)

    #### add nodes attributes ####
    print('step 2: adding node attributes')
    for x in PATENT_INFO.columns:
        G.vs[x] = array(PATENT_INFO[x])

    #### add edges ####
    print('step 3: adding edges')
    # create a series with 'patent_number' as index and ID from zero to N as values. 
    # We will then use it to translate the edgelist from patent numbers to patent IDs. We will use 'map' for that
    PATENT_INFO['ID'] = range(shape(PATENT_INFO)[0])
    series_patent_ids = PATENT_INFO[['ID','patent_number']].astype({'patent_number':str}).set_index('patent_number').iloc[:,0]
    
    # Now map patent numbers to IDs in our edgelist
    citations['citing_numerical_id'] = citations['Citing_Patent'].astype(str).map(series_patent_ids)
    citations['cited_numerical_id'] = citations['Cited_Patent'].astype(str).map(series_patent_ids)
    
    edgelist = citations[['cited_numerical_id','citing_numerical_id']].astype(int)
    print(edgelist.head(10))
    print(edgelist.describe())
    print(edgelist.info())
    # now populate our graph G with edges. We will then be able to select subgraphs based on nodes (or edges) attributes
    G.add_edges(edgelist.to_records(index=False))

    del edgelist
    gc.collect()

    return G


def topologically_sort_graph(G):
    if not G.is_dag():
        print('Graph is not DAG. If you continue the topological sorting algorithm will be stuck in an endless search. Remove cycles before to continue!')
        raise ValueError
    layers = -1 * ones(G.vcount())
    nodes_in_this_layer = where(array(G.indegree())==0)[0]
    layer = 0

    while nodes_in_this_layer.any():
        layers[nodes_in_this_layer] = layer
        layer += 1
        nodes_in_this_layer = G.neighborhood(vertices=nodes_in_this_layer.tolist(), order=1, mode='OUT')
        nodes_in_this_layer = unique([item for sublist in nodes_in_this_layer for item in sublist[1:]])
    return layers


def search_path_count_of_graph(G, mode='IN', layer_name='layer'):
    layers = G.vs[layer_name]
    if mode=='IN':
        layer_values = arange(2,max(layers)+1)
    elif mode=='OUT':
        layer_values = arange(max(layers)-2, -1, -1)
    count_paths = array(G.degree(mode=mode))
    for layer in layer_values:
        for n in where(layers==layer)[0]:
            neighbors = G.neighbors(n, mode=mode)
            if neighbors:
            #Each node's count of incoming paths is the sum of its predecessors' count of incoming paths
                count_paths[n] += sum(count_paths[array(neighbors)])
    #Added absolute value to remove negative values
    return np.abs(count_paths).astype(np.float32)


def randomize_citations(citations,
                        patent_attributes):
    citations_randomized = citations.copy()
    not_same_year = citations_randomized['Year_Citing_Patent']!=citations_randomized['Year_Cited_Patent']
    # Take the same-class citations of every class and permute them.
    print("Randomizing Same-Class Citations")

    citations_randomized.rename(columns={"Class_citing":"Class_Citing_Patent","Class_cited":"Class_Cited_Patent"},inplace=True)
    same_class_ind = citations_randomized['Class_Citing_Patent']==citations_randomized['Class_Cited_Patent']
    cross_class_ind = -same_class_ind 
    same_class_ind = same_class_ind & not_same_year
    grouper = citations_randomized.loc[same_class_ind].groupby(['Year_Citing_Patent','Year_Cited_Patent', 'Class_Citing_Patent',])[['Citing_Patent',    'Cited_Patent']]
    print("%i groups"%(len(grouper)))
    print("%i groups that can be rewired"%(sum(grouper.size()>1)))
    g = grouper.apply(randomize_citations_helper)
#    g.index = g.index.droplevel(['Year_Citing_Patent','Year_Cited_Patent','Class_Citing_Patent'])

    citations_randomized.loc[same_class_ind,['Citing_Patent', 'Cited_Patent']] = g

    ### Take the cross-class citations and permute them.
    print("Randomizing Cross-Class Citations")        
    cross_class_ind = cross_class_ind & not_same_year
    grouper = citations_randomized.loc[cross_class_ind].groupby(['Year_Citing_Patent','Year_Cited_Patent',])[['Citing_Patent','Cited_Patent']]
    print("%i groups"%(len(grouper)))
    print("%i groups that can be rewired"%(sum(grouper.size()>1)))
    g = grouper.apply(randomize_citations_helper)
#     g.index = g.index.droplevel(['Year_Citing_Patent','Year_Cited_Patent'])

    citations_randomized.loc[cross_class_ind, ['Citing_Patent', 'Cited_Patent']] = g
    
    ### Drop patent attributes (which are now inaccurate for both the citing and cited patent) and bring them in from patent_attributes
    citations_randomized.drop(['Class_Citing_Patent', 'Class_Cited_Patent'], axis=1, inplace=True)
#     citations_randomized = citations_randomized[['Citing_Patent', 'Cited_Patent', 'Same_Class']]

    patent_attributes = patent_attributes[['patent_number', 'Class']].set_index('patent_number')
    citations_randomized = citations_randomized.merge(patent_attributes,left_on='Citing_Patent',right_index=True,)

    citations_randomized = citations_randomized.merge(patent_attributes, 
                    left_on='Cited_Patent', 
                    right_index=True,
                    suffixes=('_Citing_Patent','_Cited_Patent'))
    print(citations_randomized.info())
    return citations_randomized

def randomize_citations_helper(citing_cited):

#     if all(citing_cited['Year_Citing_Patent']==citing_cited['Year_Cited_Patent']):
#         return citing_cited[['Citing_Patent', 'Cited_Patent']]
    n_Citing = citing_cited.Citing_Patent.nunique()
    n_Cited = citing_cited.Cited_Patent.nunique()
    # print(n_Citing,n_Cited,citing_cited.shape[0])
    # if n_Cited*n_Citing==citing_cited.shape[0]: #Original code, changed to pass BiRewire log(negative) math domain error
    if n_Cited*n_Citing<=citing_cited.shape[0] or (n_Cited*n_Citing)==0: #The graph is fully connected, and so can't be rewired
        return citing_cited#[['Citing_Patent', 'Cited_Patent']]
    
#     Citing_lookup = pd.Series(index=citing_cited.Citing_Patent.unique(),
#                               data=1+arange(n_Citing))
#     Cited_lookup = pd.Series(index=citing_cited.Cited_Patent.unique(),
#                              data=1+arange(n_Cited))
#     input_to_Birewire = array([Citing_lookup.ix[citing_cited.Citing_Patent].values,
#                                Cited_lookup.ix[citing_cited.Cited_Patent].values + n_Citing]).T
    citing_lookup = citing_cited['Citing_Patent'].astype('category')
    cited_lookup = citing_cited['Cited_Patent'].astype('category')
    input_to_Birewire = array([citing_lookup.cat.codes.values.astype('uint64'),
                               cited_lookup.cat.codes.values.astype('uint64') + n_Citing]).T+1
#     citing_cited.Citing_Patent = Citing_lookup.ix[citing_cited.Citing_Patent].values
#     citing_cited.Cited_Patent = Cited_lookup.ix[citing_cited.Cited_Patent].values
#     citing_cited.Cited_Patent += n_Citing
    this_rewiring = BiRewire.Rewiring(data=input_to_Birewire,
                               type_of_array='edgelist_b',
                               type_of_graph='bipartite')
    this_rewiring.rewire(verbose=0)   
    z = this_rewiring.data_rewired-1

#     Citing_lookup = pd.DataFrame(Citing_lookup).reset_index().set_index(0)
#     Cited_lookup = pd.DataFrame(Cited_lookup).reset_index().set_index(0)
#     citing_patents = Citing_lookup.ix[z[:,0]].values.flatten()
#     cited_patents = Cited_lookup.ix[z[:,1]-n_Citing].values.flatten()
    
    citing_patents = citing_lookup.cat.categories.values[z[:,0]]
    cited_patents = cited_lookup.cat.categories.values[z[:,1]-n_Citing]

    rewired_output = pd.DataFrame(index=citing_cited.index,
                                 columns=['Citing_Patent', 'Cited_Patent']
                                  )
    if len(citing_cited.index)!=len(citing_patents):
        # print(citing_cited.index,citing_patents)
        print("Citing_cited index != citing_patents Error")
        return citing_cited
    else:
        rewired_output['Citing_Patent'] = citing_patents
        rewired_output['Cited_Patent'] = cited_patents
    return rewired_output

################
# Data imports #
################
print(f"248: Importing data {randomization_id}")

patents = pd.read_csv(data_input+class_system+'/'+class_system+'_patents.csv',low_memory=False,usecols=["patent_number","filing_year","patent_id","Class"])
#YEAR FLTER
patents=patents[patents['filing_year']<=LAST_YEAR]
# patents=patents[(patents['filing_year']>=FIRST_YEAR) & (patents['filing_year']<=LAST_YEAR)]
# patents = pd.read_csv(data_input+class_system+'/'+class_system+'_patents.csv',low_memory=False,usecols=["patent_number","filing_year","patent_id","Class"]).replace('', np.nan).dropna()
patents=patents.astype({"patent_number":"str"})

print(class_system+"patents.csv import done")
print(f"Description of {class_system} patents:")
print(patents.info())
print(patents.describe())

mylist=[]
for chunk in pd.read_csv(data_input+class_system+'/'+class_system+'_citations.csv',low_memory=False,chunksize=200000,usecols=["Cited_Patent","Citing_Patent","Year_Cited_Patent","Year_Citing_Patent","Class_citing","Class_cited" ]):
    mylist.append(chunk)
# citations = pd.read_csv(data_input+class_system+'/'+class_system+'_citations.csv',low_memory=False).replace('', np.nan).dropna()
citations=pd.concat(mylist,axis=0)
# citations_made=citations["Citing_Patent"].value_counts().to_frame()
del mylist
print(class_system+"citations.csv import done")
print(f"Description of {class_system} citations:")
print(citations.info())
print(citations.describe())

# Completeness filter, just take citations where both patents are in patents dataframe
# def Completeness_filter(citations,patents):
citations=citations.astype({"Cited_Patent":"str","Citing_Patent":"str"})
citations=citations[(citations["Cited_Patent"].isin(patents['patent_number'])) & (citations["Citing_Patent"].isin(patents['patent_number']))]
# Completeness_filter(citations,patents)
print("272: Imports done after {} minutes".format((time.time()-start_time)/60))

if randomized_control:
    ti = time.time()
    citations = randomize_citations(citations, patents)
    tf = time.time()
    final_time_length = tf-ti
    print('282: Done! Randomizing citations took: %f seconds' %final_time_length + '= %f minutes' %(final_time_length/60))
    with open(data_output+class_system+f'/Controls/Synthetic_controls/{FIRST_YEAR}{LAST_YEAR}/Ready.txt',"a") as MyFile:
        MyFile.write(f"\n {randomization_id}: Randomization done") 

##################################
# Creating patent citation graph #
##################################)
ti = time.time()
G = create_patent_citation_graph(patents, citations)
if not G.is_dag():
    print('289: Graph is not DAG. If you continue the topological sorting algorithm will be stuck in an endless search. Remove cycles before to continue!')
    raise ValueError
print('291: Graph is DAG, you can continue!')
del patents #Clears RAM space
gc.collect()
layers = topologically_sort_graph(G)
G.vs['layer'] = layers
print(f"296: Maximum number of layers:{max(layers)}")

count_incoming_paths = array(search_path_count_of_graph(G, mode='IN'))
G.vs['count_incoming_paths'] = count_incoming_paths

tf = time.time()
time_length = (tf-ti)/60 # unit = minutes
print("314: Done creating patent-citation graph! Elapsed time: %f minutes" %time_length) # takes 8.2 minutes to run

#Number of patents until each year (year is index)
vc = pd.value_counts(G.vs['filing_year']).sort_index()
# n_rows_support=vc[vc.index>=FIRST_YEAR].cumsum() #If this works, G.vs.select(filing_year_le=observation_year) selects patents only filed at FIRST_YEAR into the subgraph
n_rows_support=vc.cumsum()[vc.index>=FIRST_YEAR] #If this works, G.vs.select(filing_year_le=observation_year) selects patents before FIRST_YEAR into the subgraph
n_rows =n_rows_support.sum()

print("320: n_rows={}".format(n_rows))

#Creating empty dataframe with enough rows for patents to be reported every year 
DF = pd.DataFrame(index=arange(n_rows), columns=["patent_number", "observation_year", "outgoing_path_count+1", "incoming_path_count+1","filing_year"])

size_in_GBs = (prod(DF.shape)*64)*1.25e-10
print("326: Memory allocated to DF_node_SPNP_over_time: %f GBs\n" %size_in_GBs)

year_list = arange(FIRST_YEAR,max(G.vs['filing_year'])+1)  
# year_list = arange(min(G.vs['filing_year']),max(G.vs['filing_year'])+1)  
this_year_data_start = 0 #Not actually a year but the index location in dataframe to start
number=0
old_row=0
for observation_year in year_list:
    
    patents_within_this_year = G.vs.select(filing_year_le=observation_year).indices
    G_subgraph = G.subgraph(patents_within_this_year, implementation="auto")
    #n_row: number of rows in this year
    n_row = G_subgraph.vcount()
    number+=n_row
    
    if old_row != n_row:
        DF.loc[this_year_data_start:this_year_data_start+n_row-1, 'outgoing_path_count+1'] = search_path_count_of_graph(G_subgraph, mode='OUT')+1.0
        DF.loc[this_year_data_start:this_year_data_start+n_row-1, 'incoming_path_count+1'] = search_path_count_of_graph(G_subgraph, mode='IN')+1.0
        DF.loc[this_year_data_start:this_year_data_start+n_row-1, 'patent_number'] = G_subgraph.vs['patent_number'].copy()
        DF.loc[this_year_data_start:this_year_data_start+n_row-1, 'filing_year'] = G_subgraph.vs['filing_year'].copy()
        DF.loc[this_year_data_start:this_year_data_start+n_row-1, 'observation_year'] = observation_year

        #Calculation of SPNP left out blank as it is calculated later on
        if observation_year%10.0==0.0:
            # print(DF.loc[this_year_data_start:this_year_data_start+n_row-1,['observation_year','outgoing_path_count_log','incoming_path_count_log']].head(5))
            # print(DF.loc[this_year_data_start:this_year_data_start+n_row-1,['observation_year','outgoing_path_count_log','incoming_path_count_log','patent_number']].tail(5))
            print("Done taking log of incoming paths for year {} with rows {} and total number {}".format(observation_year,this_year_data_start-this_year_data_start+n_row-1,number))
    old_row=n_row
    this_year_data_start += n_row
    del G_subgraph
    gc.collect()

DF['observation_year'] = DF['observation_year'].astype('float')
DF['patent_number'] = DF['patent_number'].astype('str')
DF['outgoing_path_count_log'] = DF['outgoing_path_count+1'].astype(np.float32).apply(np.log)
DF['incoming_path_count_log'] = DF['incoming_path_count+1'].astype(np.float32).apply(np.log)
DF['SPNP_count'] = DF['outgoing_path_count+1']*DF['incoming_path_count+1']
print(DF.describe())
print(DF.info())
print(DF.isna().sum())

print("368: SPNP count done done after {} minutes".format((time.time()-start_time)/60))

# # Report patents' centrality in t+2, t+3, t+5 and t+8
#Patents on last observation, all in 2020
DF_patents = DF[DF['observation_year']==LAST_YEAR].copy() 
DF.drop(columns=['outgoing_path_count+1','incoming_path_count+1'],inplace=True)
#"patent_number",  "SPNP_count", "filing_year", "outgoing_path_count_log", "incoming_path_count_log",
DF_patents.drop(columns=['observation_year'], axis=1, inplace=True)

del DF['outgoing_path_count_log']
gc.collect()

ti = time.time()
# for patents filed AFTER 1975 report their centrality 3/5/8 years after filing
# for patents filed BEFORE 1975 report their centrality in 1978/1980/1983. To quickly do this create a "fake filing year"
DF_patents['fake_filing_year'] = DF_patents['filing_year']
DF_patents.loc[DF_patents['filing_year']<FIRST_YEAR, 'fake_filing_year']=FIRST_YEAR/1.0


for horizon in [2,3,5,8]:
    DF_patents[f'filing_year+{horizon}'] = DF_patents['fake_filing_year']+horizon
    # merge to add SPNP after $horizon years from filing
    DF_patents = pd.merge(DF_patents, DF, how='left', left_on=['patent_number',f'filing_year+{horizon}'], right_on=['patent_number','observation_year'], suffixes=('', f'_{horizon}'))
    del DF_patents[f'filing_year_{horizon}']
    del DF_patents['observation_year']
    gc.collect()
#Should have as NANS the number of patents with no records after {horizon} years:
#h=2: 13229388 = 6612159+6617229
#h=3: 19779109 = 6549721+6612159+6617229
#h=5: 32374008 = 6549721+6612159+6617229+6404073+6190826
#h=8: 49441092 = 6549721+6612159+6617229+6404073+6190826+5946975+5689553+5430556

del DF_patents['fake_filing_year']
gc.collect()

tf = time.time()
final_time_length = tf-ti
print('404: Done! Reporting patent centrality after '+str(LAST_YEAR-FIRST_YEAR)+' years took: %f seconds' %final_time_length + '= %f minutes' %(final_time_length/60))
print("---------------------------- DF_patents -----------------------------")
print(f"DF_patents is the set of patents obsered at {LAST_YEAR}")
print(DF_patents.columns)
print(f'DF_patents has columns: {DF_patents.columns}')
DF_patents.rename(columns={'SPNP_count': 'SPNP_count_'+str(LAST_YEAR),'outgoing_path_count_log': 'outgoing_path_count_log_'+str(LAST_YEAR),'incoming_path_count_log': 'incoming_path_count_log_'+str(LAST_YEAR)}, inplace=True)
print(DF_patents.describe())
print(DF_patents.info())
print(DF_patents.isna().sum())

################################################## 
#   Compute centrality of cited patents in t-1   #
##################################################

#First: we add SPNP count of cited patents for the year it was cited
citations = pd.merge(citations, DF,how='left', on=None, left_on=['Cited_Patent','Year_Citing_Patent'], right_on=['patent_number','observation_year'])
del DF['incoming_path_count_log']
del citations['patent_number']
del citations['observation_year']
gc.collect()
citations.rename(columns={"SPNP_count": 'SPNP_count_cited_year_of_citation'}, inplace=True)

#Second: Citations are grouped by citing patent, new columns containing mean SPNP and the std of cited patents at the citation year
citations_grouped_by_citing = citations[['SPNP_count_cited_year_of_citation', 'Citing_Patent']].groupby(['Citing_Patent']).agg({'SPNP_count_cited_year_of_citation':["mean","std"]})
citations_grouped_by_citing.columns = citations_grouped_by_citing.columns.droplevel(0)
citations_grouped_by_citing.reset_index(inplace=True)

#Third: the new data frame is added to DF_patents. MeanSPNPcited and stdSPNP cited correspond to the SPNP count of cited patents by each patent at the year of filing
DF_patents = pd.merge(DF_patents, citations_grouped_by_citing,how='left', left_on='patent_number',right_on='Citing_Patent')

#Printing results and collecting garbage
print("\n435: citations done")
#Contains Nans

#Should include column meanSPNPcited and stdSPNPcited
print("\n 439: DF_patents")
#Fourth: clean up for next mean and std
DF_patents.rename(columns={"mean": 'meanSPNPcited', "std": 'stdSPNPcited'}, inplace=True)


#Fifth: we add to citations new SPNP count, one year before citing takes place
citations['Year_Citing_Patent-1'] = citations['Year_Citing_Patent'] - 1
citations = pd.merge(citations, DF ,how='left', on=None, left_on=['Cited_Patent','Year_Citing_Patent-1'], right_on=['patent_number','observation_year'])
del DF
del citations['patent_number']
del citations['observation_year']
del citations['Year_Citing_Patent-1']
gc.collect()

citations.rename(columns={"SPNP_count": 'SPNP_count_cited_1year_before_citation'}, inplace=True)

#Sixth: Some patents are cited the same year they were filed. For these patents we have no information on their SPNP the year before they were cited. We need to compute their SPNP the moment before they were cited. This is done by multiplying the number of incoming paths of the cited patent by the number of citations received in the year they were filed. This is only an approximation of the number of their outgoing paths, but it is not a bad one because they are unlikely to be cited by other patents granted in the same year that have received citations themselves.
citations_same_year_ind = citations['Year_Citing_Patent']==citations['Year_Cited_Patent']
citations_same_year_count = citations[citations_same_year_ind].groupby(['Cited_Patent']).size()
citations_same_year_count.name = 'citations_at_zero'
citations_same_year = pd.DataFrame(citations_same_year_count).reset_index() #DF with Cited_Patent and citations_at_zero


citations_same_year = pd.merge(citations_same_year, DF_patents[['incoming_path_count_log_'+str(LAST_YEAR),"patent_number"]].astype({'incoming_path_count_log_'+str(LAST_YEAR):'float32'}),  how='left', left_on=["Cited_Patent"], right_on=["patent_number"]).drop(columns="patent_number")

#Seventh: Approximation is made for patents cited the year they were filed and added to citations Calculates log(SPNP) 1 year before citation next line is without log!
citations_same_year['SPNP_at_Year_Cited_Patent'] = citations_same_year['incoming_path_count_log_'+str(LAST_YEAR)].fillna(0)+log1(citations_same_year['citations_at_zero']+1)
# citations_same_year['SPNP_at_Year_Cited_Patent'] = citations_same_year['incoming_path_count_log_'+str(LAST_YEAR)].fillna(0).apply(np.exp)*(citations_same_year['citations_at_zero']+1)
# citations_same_year has columns  Cited_Pated,citations_at_zero,incoming_path_count_log,patent_number,SPNP_at_Year_Cited_Patent

citations1=pd.merge(citations.loc[citations['Year_Citing_Patent']==citations['Year_Cited_Patent']].drop(columns=['SPNP_count_cited_1year_before_citation']),citations_same_year[['SPNP_at_Year_Cited_Patent','Cited_Patent']],how='left',right_on='Cited_Patent',left_on='Cited_Patent').rename(columns={'SPNP_at_Year_Cited_Patent':'SPNP_count_cited_1year_before_citation',"filing_year_x":"filing_year"}).drop(columns=['filing_year_y'])
#Garbage collection
del citations_same_year_ind
del citations_same_year
del citations_same_year_count
gc.collect()

citations=citations.loc[citations['Year_Citing_Patent']!=citations['Year_Cited_Patent']].drop(columns=["filing_year_y"]).rename(columns={"filing_year_x":"filing_year"})
citations=pd.concat([citations,citations1])
del citations1
gc.collect()

print("481: Approximation for same year citations done")
#############################
# Final merge to DF_patents #
#############################
#Another DF with the mean and std SPNP count of cited patents one year before citations are made per citing patent 
citations_grouped_by_citing = citations.loc[:,['SPNP_count_cited_1year_before_citation','Citing_Patent']].groupby('Citing_Patent').agg(['mean','std'])
citations_grouped_by_citing.columns=citations_grouped_by_citing.columns.droplevel(0)
citations_grouped_by_citing.reset_index(inplace=True)

DF_patents = pd.merge(DF_patents, citations_grouped_by_citing, how='left', left_on='patent_number', right_on='Citing_Patent')
DF_patents.rename(columns={"mean": 'meanSPNPcited_1year_before',"std": 'stdSPNPcited_1year_before'}, inplace=True)

# exit()
print("citations has ")
cols_drop=['Cited_Patent','Year_Cited_Patent','Year_Citing_Patent','Class_Citing_Patent','Class_Cited_Patent','Class_citing','Class_cited','citing_numerical_id','cited_numerical_id','SPNP_count_cited_year_of_citation','filing_year','SPNP_count_cited_1year_before_citation']
for col in cols_drop:
    try:
        citations.drop(columns=[col],inplace=True)
    except Exception:
        print(f"The column {col} was dropped already!(or wasnt found)")
# cols_drop=['Cited_Patent','Year_Cited_Patent','Year_Citing_Patent','Class_Citing_Patent','Class_Cited_Patent','citing_numerical_id','cited_numerical_id','SPNP_count_cited_year_of_citation','filing_year','SPNP_count_cited_1year_before_citation']
# citations.drop(columns=cols_drop,inplace=True)
# citations=citations[['Citing_Patent','incoming_path_count_log']] #Deberia funcionar igual
gc.collect()
citations.rename(columns={"incoming_path_count_log":'incoming_path_count_log_cited'},inplace=True)

#Another DF with the mean and std incoming path count of cited patents one year before citations are made per citing patent 
citations_grouped_by_citing = citations.groupby('Citing_Patent').agg({'incoming_path_count_log_cited':['mean','std']})
citations_grouped_by_citing.columns =citations_grouped_by_citing.columns.droplevel(0) #Filtro de drop level
citations_grouped_by_citing.reset_index(inplace=True)

del citations 
gc.collect()

DF_patents = pd.merge(DF_patents, citations_grouped_by_citing, how='left', left_on='patent_number', right_on='Citing_Patent')
DF_patents=DF_patents.rename(columns={"mean": 'mean_incoming_path_count_log_cited',"std": 'std_incoming_path_count_log_cited'}).drop(columns=["Citing_Patent_x","Citing_Patent_y"])
DF_patents['filing_year'] = DF_patents['filing_year'].astype("float").astype('uint16')
DF_patents['patent_number'] = DF_patents['patent_number'].astype("str")

# Applying_logs #Order of operations to confirm: if its SPNP-log-Randomization-mean or SPNP-Randomization-mean-log
# for col in ['SPNP_count_3',',SPNP_count_5','SPNP_count_2','SPNP_count_5',f'SPNP_count_{LAST_YEAR}','meanSPNPcited','stdSPNPcited','meanSPNPcited_1year_before','stdSPNPcited_1year_before']:
#     DF_patents['log_'+col]=DF_patents[col].apply(log1)

for c in DF_patents.columns:
    if DF_patents[c].dtype =='float':
        DF_patents[c] = DF_patents[c].astype('float32')
        
print("\n528: DF_patents that will be exported")
print(DF_patents.sample(10))
print(DF_patents.info(memory_usage="deep"))
print(DF_patents.isna().sum())

if not randomized_control:
    # DF_patents.to_hdf(data_directory+'centralities/empirical.h5', 'df', complevel=9, complib='blosc')
    DF_patents.to_csv(data_output+class_system+'/'+class_system+f'_empirical_{FIRST_YEAR}_{LAST_YEAR}.csv')
else:
    # DF_patents.to_hdf(data_directory+'centralities/controls/%s/synthetic_control_%i.h5'%(class_system,randomization_id), 'df', complevel=9, complib='blosc')
    DF_patents.to_csv(data_output+class_system+f'/Controls/Synthetic_controls/{FIRST_YEAR}{LAST_YEAR}/synthetic_control_{randomization_id}.csv')
    with open(data_output+class_system+f'/Controls/Synthetic_controls/{FIRST_YEAR}{LAST_YEAR}/Ready.txt',"a") as MyFile:
        MyFile.write(f"\n {randomization_id}: File done") 

final_time_length = time.time()-start_time

print('\nDone! Total job took: %f seconds' %final_time_length + '= %f minutes' %(final_time_length/60))