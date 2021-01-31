import os
import numpy as np
list_par = []
cluster_list_par= []
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def load_data(folder_par):
    for count,filename in enumerate(os.listdir(os.path.join( folder_par))):
        label  = filename[:-6].replace("_","")
        cluster_list_par.append((np.loadtxt(os.path.join(folder_par,filename))))
        list_par.append((np.asarray([label,len([i for i in list_par if i[0][0]==label]),]*(cluster_list_par[count].shape[0]))))     
        list_par[count] = np.reshape(list_par[count], (int(len(list_par[count])/2),2)   )
        cluster_list_par[count]= np.hstack((list_par[count],cluster_list_par[count])) 

    return np.vstack(cluster_list_par)

def get_coded_observations(clustering_array_par):
    
    # extract unique text labels
    class_text = np.unique(clustering_array_par[:,0])

    # deep copy input array
    raw_data_par = np.copy(clustering_array_par)
    raw_data_text = np.copy(clustering_array_par)
    
    # transform motion class into 0-4 codeword
    indexer = []
    for i,j in enumerate(class_text):
        indexer.append(np.where(raw_data_par[:,0] == j))
        
        raw_data_par[:,0][indexer[-1]] = int(i)
        raw_data_text[:,0][indexer[-1]] = j
    return raw_data_par,raw_data_text


def rescale_data(data_array_par):
    
    # create empty lists
    ob_split = []
    ob_split_scaled = []
    
    # for all coded data 
    for i in np.unique(data_array_par[:,0]):
        
        # extract elements for each code (i.e.: beat3 = 0 and so on)
        j = data_array_par[data_array_par[:,0]==i]
        
        # for all the datapoints that belong to a given observation (unique elements of column 1)
        for j1 in np.unique(j[:,1]):
            
            # get the time samples in float format
            k = j[j[:,1]==j1].astype(float)
            
            # append single list element containing all time samples
            ob_split.append(k)
    
    for cnt, observation in enumerate(ob_split):
        
        # copy entire array, reset set type to float
        deep_ob_copy = np.copy(observation).astype(float)
        
        # extract min and max feature values
        dim_max = np.amax(deep_ob_copy[:,3:],axis=0)
        dim_min = np.amin(deep_ob_copy[:,3:],axis=0)
        
        # rescale features
        deep_ob_copy[:,3:] = (deep_ob_copy[:,3:]-dim_min)/(dim_max-dim_min)
        
        # append to scaled list
        ob_split_scaled.append(deep_ob_copy)
        
        # if first iteration, put the first stack of observations in array
        if cnt == 0:            
            ob_rescaled = deep_ob_copy
            
        # else stack subsequent sets of observations in array
        else:
            ob_rescaled = np.vstack((ob_rescaled,deep_ob_copy))
    
    # return relevant variables
    return ob_rescaled, ob_split,ob_split_scaled