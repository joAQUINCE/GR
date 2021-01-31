# import built-in modules
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

# import custom modules
import dataloader
import lambdaloader
custom_modules = ["dataloader", "lambdaloader"]
for module_par in custom_modules:
    del sys.modules[module_par]
    
import dataloader
import lambdaloader
# import all files in "train"
clustering_array = dataloader.load_data("train")
# set recursion depth higher
sys.setrecursionlimit(3500)

labels_all_train    = np.asarray(clustering_array[:,:2])
features_all_train  = clustering_array[:,3:]

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

coded_data,coded_data_text = get_coded_observations(clustering_array)
coded_data_number = coded_data.astype(float)

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

features_normalized, observation_list, observation_list_scaled = rescale_data(coded_data)

# PLOT DATA IN 2D
start = 0
ending   = 10
point_cnt = 3000
fig, ax = plt.subplots((ending-start),3, figsize =(18,42),constrained_layout=True)
for i in range(start,ending,1):
    for ind,j in enumerate([[3,4,"X Linear Acceleration (0:1 Normalized)","Y Linear Acceleration (0:1 Normalized)"],                            [3,5,"X Linear Acceleration (0:1 Normalized)","Z Linear Acceleration (0:1 Normalized)"],                            [4,5,"Y Linear Acceleration (0:1 Normalized)","Z Linear Acceleration (0:1 Normalized)"]]):
        ax[i-start,ind].scatter(observation_list_scaled[i][:point_cnt,j[0]],observation_list_scaled[i][:point_cnt,j[1]])
        ax[i-start,ind].set_xlabel(j[2])
        ax[i-start,ind].set_ylabel(j[3])
fig.suptitle('Linear Acceleration Plots (0:1 Normalized)', fontsize=35)
fig.savefig("Linear Acceleration Plots (0-1 Normalized).png")
plt.show()

def find_kmeans_center(indices_list_par,k_means_runs_par):
    
    similarity_array= np.zeros([k_means_runs_par,k_means_runs_par,cluster_cnt])
    
    for run_id1,indices_set1 in enumerate(indices_list_par):
        old_time = time.time()
        for run_id2,indices_set2 in enumerate(indices_list_par):
            if run_id1 != run_id2:
                for current_indices_list1 , i in enumerate(indices_set1):
                    list_comp = []
                    for current_indices_list2 , j in enumerate(indices_set2):
                        list_comp.append(len(set(i[0][0])&set(j[0][0])))
                    similarity_array[run_id1,run_id2,current_indices_list1]=max(list_comp)
        print("Centers for Run", run_id1, "completed in",  time.time()-old_time,"seconds.")
    similarity_count_final = np.sum(similarity_array,axis=2)
    return np.argmax(similarity_count_final) % (similarity_count_final.shape[1])

def get_kmeans(input_data, convergence_crit):
    
   
    k_means_runs = 50
    centers_list = []
    center_assignment_list = []
    indices_list =[]
    for run_count in range(k_means_runs):
        old_time = time.time()
        centers_init = np.random.rand(cluster_cnt, 6)*.10 +.45
        centers_init=centers_init.astype(np.float)
        input_data_par = input_data.astype(np.float)
        center_assignments = np.zeros(len(input_data_par[:,1]),dtype="int")
        kmeans_centers = np.copy(centers_init)
        oldcenters = np.copy(centers_init)
        centers_shift = 10 #arbitrary initial value greater than convergence_crit
        iterations   = 0  #initialize iteration counter
        while centers_shift > convergence_crit and iterations < 100:
            indices = []
            # create row indexer
            data_indexer   = np.arange(input_data_par.shape[0])[:,np.newaxis] #generate array of row indices
            # create column indexer
            center_indexer = np.transpose(np.arange(centers_init.shape[0])[:,np.newaxis]) #generate array of column indices
            # find distances to each center
            point_dists = np.linalg.norm(input_data_par[data_indexer] - kmeans_centers[center_indexer],axis = 2)
            # iteration counter
            iterations = iterations + 1
            #Update cluster assignments
            for k in range(len(input_data_par[:,1])):
                center_assignments[k] = np.argmin(point_dists[k])
            #Update kmeans_centers
            for m in range(len(init_centers)):
                indices.append([np.where(center_assignments == m)])
            for index,n in enumerate(indices):
                if len(n[0][0]) > 0:                
                    kmeans_centers[index] = (np.mean(np.squeeze(input_data_par[n]),axis = 0)) 
            #Check for convergence
            update_array = np.sqrt(np.sum((kmeans_centers-oldcenters) **2, axis=1))
            centers_shift = np.mean(update_array)
            
            #if average of distance kmeans_centers moved is less than some tolerance, terminate
            if (centers_shift > convergence_crit): 
                oldcenters  =  np.copy(kmeans_centers)
            else:
                print("The K-means Algorithm has converged for run", run_count+1,"of",k_means_runs," planned runs after", time.time()-old_time,"seconds and",                     iterations,"iterations.")
                center_assignment_list.append(center_assignments)
                centers_list.append(kmeans_centers)
                indices_list.append(indices)
                # if first iteration, put the first stack of observations in array
                new_count_stack = run_count*np.ones(kmeans_centers.shape[0])
                if run_count == 0:            
                    
                    center_stack = np.sort(kmeans_centers,axis = 1)
                    iteration_count_stack = new_count_stack
                # else stack subsequent sets of observations in array
                else:
                    center_stack =           np.vstack((center_stack,np.sort(kmeans_centers,axis = 1)))
                    iteration_count_stack =  np.hstack((iteration_count_stack,new_count_stack))
                break
    
    correct_centers_iteration = find_kmeans_center(indices_list,k_means_runs)       
    
    kmeans_centers = centers_list[correct_centers_iteration]
    clusters = center_assignment_list[correct_centers_iteration]
    
    np.savetxt("kmeans_output/" +str(cluster_cnt)+"_centers.txt",kmeans_centers)
    np.savetxt("kmeans_output/" +str(cluster_cnt)+"_center_loc_train.txt",clusters)
    
    return kmeans_centers, clusters

#Initialize centers and tolerance and run K-Means
cluster_cnt = 40
tolerance = .005
init_centers = np.random.rand(cluster_cnt, 6)*.20 +.4
kmeans_centers, clusters = get_kmeans(features_normalized[:,-6:], tolerance)

vector_quantized_data = np.hstack((coded_data_number[:,0:2],clusters[:,np.newaxis])).astype(np.int)
label_quantized_data = np.hstack((coded_data_text[:,0:2],clusters[:,np.newaxis]))

def recursive_alpha(time_steps_cnt_par, state_cnt_par, alpha_matrix_par,               A_matrix_of_ob_par, B_matrix_of_ob_par, time_recursion_shifter_par,c_populate_matrix_par,single_ob_par,rec_par):
    
    rec = rec_par + 1 
    recursion_shifter_par = time_recursion_shifter_par
    
    # create deep copy of matrix_par
    alpha_recursion_mat = np.copy(alpha_matrix_par)   
       
    # create c_populate matrix of size T-1 to populate values 1:T of the c_t matrix
    c_populate_matrix = np.copy(c_populate_matrix_par)
    
    if recursion_shifter_par <= time_steps_cnt_par - 1:
        
        # extract the applicable column of b, which is found by seeing the value of the sequence at time t
        # this will be one of the possible values, in this case, a cluster center label
        current_b_matrix_column = single_ob_par[recursion_shifter_par]
        alpha_recursion_mat[recursion_shifter_par,:] =  np.dot(alpha_matrix_par[recursion_shifter_par-1,:],A_matrix_of_ob_par) 
        alpha_recursion_mat[recursion_shifter_par,:] =  np.multiply(alpha_recursion_mat[recursion_shifter_par,:],
                                                                   B_matrix_of_ob_par[:,current_b_matrix_column])
        # polulate one element [recursion_shifter_par] of c_populate_matrix
        c_populate_matrix  [recursion_shifter_par] =  np.sum(alpha_recursion_mat[recursion_shifter_par,:])
        
        # do one over c_populate for scale
        c_populate_matrix[recursion_shifter_par] = 1/ (c_populate_matrix[recursion_shifter_par])
        
        
        # rescale alpha_recursion_mat[recursion_shifter_par,:]
        # pass to recursion!!
        alpha_recursion_mat[recursion_shifter_par,:] = c_populate_matrix [recursion_shifter_par] * alpha_recursion_mat[recursion_shifter_par,:]
        
        # add one to the downwards shifter so we index the row above on next recursion
        # pass to recursion!!
        recursion_shifter_new = recursion_shifter_par + 1
        
        alpha_mat_temp,c_populate_matrix_temp = recursive_alpha(time_steps_cnt_par, state_cnt_par, alpha_recursion_mat,                                       A_matrix_of_ob_par, B_matrix_of_ob_par, recursion_shifter_new,c_populate_matrix,                                                              single_ob_par,rec)
        
        c_populate_matrix   = c_populate_matrix_temp
        alpha_recursion_mat = alpha_mat_temp 
        
    return alpha_recursion_mat,c_populate_matrix

def get_alpha_matrix(single_ob_par, A_matrix_of_ob_par, B_matrix_of_ob_par, N, pi_par):
    
    # extract number of time steps
    time_steps_cnt_par = single_ob_par.shape[0]
    # extract number of states 
    state_cnt_par = B_matrix_of_ob_par.shape[0]
    # create c (scaling factor matrix)
    c_scaling_factor = np.zeros([time_steps_cnt_par])
    
    # perform dimension test. if failed, break, print and exit if wrong. 
    if state_cnt_par != A_matrix_of_ob_par.shape[0] or state_cnt_par != A_matrix_of_ob_par.shape[1]:
        return print("Dimensions of A and B are incorrect - CODE NOT EXECUTED")   
    
    else:
        alpha_matrix_par = np.zeros([time_steps_cnt_par,state_cnt_par])
        
        # extract the first value in the current sequence
        obsservation_zero = single_ob_par[0]
        
        # compute time-zero alphas (first Nx1 row of TxN matrix)
        alpha_matrix_par[0,:] = np.multiply(pi_par, B_matrix_of_ob_par[:,obsservation_zero]) 
        
        # compute c{t=0}
        c_scaling_factor[0]   = np.sum(alpha_matrix_par[0,:])
        # recompute c{t=0} as its inverse
        c_scaling_factor[0] = 1/c_scaling_factor[0]
        # rescale alphas at time zero
        alpha_matrix_par[0,:] = c_scaling_factor[0] * alpha_matrix_par[0,:]
        
        # this value determines the first time index determined during recursion
        time_recursion_shifter_par = 1
        
        alpha_matrix_par_temp,c_matrix = recursive_alpha(time_steps_cnt_par,state_cnt_par,alpha_matrix_par,                                              A_matrix_of_ob_par, B_matrix_of_ob_par,time_recursion_shifter_par,                                                        c_scaling_factor,single_ob_par,0)
    
    return alpha_matrix_par_temp , c_matrix

def beta_recursion(matrix_par,A_matrix_of_ob_par, B_matrix_of_ob_par,time_steps_cnt_par,recursion_shifter_par,c_par,                   single_ob_par):
    # create deep copy of matrix_par
    recursive_indexer = time_steps_cnt_par - recursion_shifter_par
    beta_recursion_mat = np.copy(matrix_par)
    
    if recursive_indexer >= 0:
        current_b_matrix_column = single_ob_par[recursive_indexer + 1]
        
        #perfom dot product from ****t+1**** parameters
        beta_recursion_mat[recursive_indexer,:] =  np.dot(beta_recursion_mat[recursive_indexer + 1,:],A_matrix_of_ob_par)
        beta_recursion_mat[recursive_indexer,:] =  c_par[recursive_indexer] * np.multiply(beta_recursion_mat[recursive_indexer,:],                                                              B_matrix_of_ob_par[:,current_b_matrix_column])
        
        # add one to the downwards shifter so we index the row above on next recursion
        recursion_shifter_new = recursion_shifter_par + 1
        # print(np.around(beta_recursion_mat, decimals = 2))
        temp = beta_recursion(  beta_recursion_mat,A_matrix_of_ob_par, B_matrix_of_ob_par,                                time_steps_cnt_par, recursion_shifter_new,c_par,single_ob_par)
        beta_recursion_mat = temp
    return beta_recursion_mat
    
def get_beta_matrix(single_ob_par, A_matrix_of_ob_par, B_matrix_of_ob_par, N_size_par, c_par):
    # extract number of time steps
    time_steps_cnt_par = single_ob_par.shape[0]
    
    # create beta matrix
    beta_matrix = np.zeros([time_steps_cnt_par,N_size_par])
    
    # assign values from C_T_minus_1_par to the last column of every matrix in the beta matrix
    beta_matrix[-1,:] = c_par[-1]
#     print("Processing time step",time_steps_cnt_par-1,"of",time_steps_cnt_par-1,"(Backward Pass)")
    
    # extract not the last element's index (shape-1) but the second to last (shape-2)
    recursion_shifter = 2
    
    # build matrix from recursive function
    beta_matrix = beta_recursion(beta_matrix,A_matrix_of_ob_par, B_matrix_of_ob_par,                                 time_steps_cnt_par,recursion_shifter,c_par,single_ob_par) 
    return beta_matrix

def get_gamma_xi_matrix(single_ob_par, A_matrix_of_ob_par, B_matrix_of_ob_par, N_size_par,alpha_par,beta_par):
    # extract number of time steps
    time_steps_cnt_par = single_ob_par.shape[0]
    
    # create gamma_xi matrix
    xi_matrix = np.zeros([time_steps_cnt_par,N_size_par,N_size_par])
    gamma_matrix    = np.zeros([time_steps_cnt_par,N_size_par])
    
    # special rule for final time step of gamma matrix
    gamma_matrix[-1,:] = alpha_par[-1,:]
       
    for time_steps in range(alpha_par.shape[0]-1):
        current_b_matrix_column = single_ob_par[time_steps + 1]
        
        for i_state in range(A_matrix_of_ob_par.shape[0]):
            for j_state in range(A_matrix_of_ob_par.shape[0]):
                xi_matrix[time_steps,i_state,j_state] = alpha_par[time_steps,i_state] *                                                           A_matrix_of_ob_par[i_state,j_state]*                                                           B_matrix_of_ob_par[j_state,current_b_matrix_column]*                                                           beta_par[time_steps+1,j_state]
    gamma_matrix[:-1] =   np.sum(xi_matrix[:-1],axis = 2)
    return xi_matrix, gamma_matrix

def get_pi(N_par):
    pi_par = 1/np.ones([N_par]) +.02*(np.random.randn(N_par))
    return pi_par

class_cnt = np.unique(label_quantized_data[:,0])
print(class_cnt)
ob_list =[]
label_list_train = []
for step_count,text_label in enumerate(class_cnt):
    class_cur = vector_quantized_data[vector_quantized_data[:,0]==step_count ]
    label_cur = label_quantized_data[label_quantized_data[:,0]==text_label ]
    
    # how many different observations there are in this unique class
    obs_in_class = len(np.unique(class_cur[:,1]))
    label_list_train.append((text_label+" ")*obs_in_class)
    
    
    for j in range(obs_in_class):
        ob    = class_cur[class_cur[:,1]==j][:,2]
        ob_list.append(ob)
        
label_list_train = (("".join(label_list_train)).strip()).split(" ")

for cnt,observation in enumerate(ob_list):
    print("Processing Observation", cnt)
    start_time = time.time()
    N_states = 5
    M_size = cluster_cnt
    pi_matrix = get_pi(N_states)
    
    a_init = 1/N_states *(np.ones([N_states, N_states]) + .1*np.random.randn(N_states, N_states))
    b_init = 1/M_size *(np.ones([N_states,M_size])  + .1*np.random.randn(N_states, M_size))
    max_iterations = 100
    current_log = -10000000
    loss_list = []
    
    for iterations in range(max_iterations):
        
        alpha, c_matrix =  get_alpha_matrix(observation, a_init , b_init, N_states, pi_matrix)
        beta = get_beta_matrix(observation, a_init, b_init, N_states,c_matrix)
        xi, gamma = get_gamma_xi_matrix(observation, a_init, b_init, N_states,alpha,beta)
        ## compute pi ##
        pi_matrix = gamma[0,:]
        ## compute new a_init ###
        num_vector   = (np.sum(xi[0:-1],axis=0))
        denom_vector = np.transpose((np.sum(gamma[0:-1],axis=0))[np.newaxis,:])
        indexer = np.arange(N_states)
        a_init = num_vector / denom_vector[indexer,:]
   
        ## compute new b_init ###
        true_arr = np.transpose(np.asarray([observation == i for i in range(b_init.shape[1])]))
        num_vector = np.dot(np.transpose(gamma),true_arr)
        denom_vector = np.sum(gamma,axis=0)[:,np.newaxis]
        indexer1 = np.arange(num_vector.shape[0])[:,np.newaxis]
        indexer2 = np.arange(M_size)[np.newaxis,:]
        b_init = num_vector[indexer1,indexer2] / denom_vector
        new_log = - np.sum(np.log(c_matrix))
        loss_list.append(new_log)
        if iterations >= max_iterations-1 or new_log < .99 * current_log: #.995 *current_log:
            print("Process Terminated at",iterations,"of",max_iterations,"iterations.")
            print("---%s Execution Time (seconds) ---" % (time.time() - start_time),"\n")
            np.savetxt("As/" +str(cnt)+".txt",a_init)
            np.savetxt("Bs/" +str(cnt)+".txt",b_init)
            np.savetxt("PIs/"+str(cnt)+".txt",pi_matrix)
            
            break
            
        current_log = new_log   

for i,j in enumerate(ob_list):
    print("Observation Number:", i)
    plt.hist(j, bins=50)
    plt.show()

import dataloader
import lambdaloader
custom_modules = ["dataloader", "lambdaloader"]
for module_par in custom_modules:
    del sys.modules[module_par]
    
import dataloader
import lambdaloader
# import all files in "train"
As, A_count   = lambdaloader.load_data("As")
Bs, B_count   = lambdaloader.load_data("Bs")
PIs, PI_count = lambdaloader.load_data("PIs")

class Lambda:
    def __init__(self,A,B,PI,motion_class,observation_number,observation,N_states,custom_ob):
        self.A = A
        self.B = B
        self.PI = PI
        self.motion_class = motion_class
        self.observation_number = observation_number
        self.observation = observation
        self.N_states = N_states
        self.custom_ob = custom_ob
    
    def get_prediction(self):
        num = get_alpha_matrix(self.custom_ob, self.A, self.B, self.N_states, self.PI)[0][-1]
        den = get_alpha_matrix(self.custom_ob, self.A, self.B, self.N_states, self.PI)[1][-1]
        rat = num/den
        return int(np.sum(rat)*100)

lambda_list = []
for ith_ob in range(A_count):
    lambda_list.append(Lambda(As[ith_ob],Bs[ith_ob],PIs[ith_ob],label_list_train[ith_ob],ith_ob,ob_list[ith_ob],N_states,ob_list[ith_ob]))

for jth_ob in range(A_count):
    lambda_list = []
    delta = 10**-100
    current_ob = np.copy(ob_list[jth_ob])
    
    for ith_ob in range(A_count):
        lambda_list.append(Lambda(As[ith_ob],Bs[ith_ob]+delta,PIs[ith_ob],"TBD",ith_ob,ob_list[ith_ob],N_states,current_ob))
    
    print("Done Setting Estimation Parameters for Observation", jth_ob,"- Performing Prediction...")
    
    pred_list = []
    pred_list = [lambda_object.get_prediction() for lambda_object in lambda_list]
    print(pred_list)
    print ("The predicted label is", label_list_train[np.argmax(np.asarray(pred_list))]+".\n")

import dataloader
custom_modules = ["dataloader"]
for module_par in custom_modules:
    del sys.modules[module_par]
    
import dataloader
clustering_array_test = dataloader.load_data("test")
coded_data_test, coded_data_text_test = get_coded_observations(clustering_array_test)
coded_data_number_test = coded_data_test.astype(float)
features_normalized_test, observation_list_test, observation_list_scaled_test = rescale_data(coded_data_test)
plt.scatter(features_normalized_test[:1000,5],features_normalized_test[:1000,6])
plt.show()

def get_cluster_assignments(test_data_par,kmeans_centers_par):
    
    # create row indexer
    data_indexer   = np.arange(test_data_par.shape[0])[:,np.newaxis] #generate array of row indices
    # create column indexer
    center_indexer = np.transpose(np.arange(kmeans_centers_par.shape[0])[:,np.newaxis]) #generate array of column indices
    # find distances to each center
    point_dists = np.linalg.norm(test_data_par[data_indexer] - kmeans_centers[center_indexer],axis = 2)
    center_assignments = np.zeros(test_data_par.shape[0],dtype="int")
       
    for k in range(test_data_par.shape[0]):
        center_assignments[k] = np.argmin(point_dists[k])
    
    return center_assignments

test_cluster_assignments = get_cluster_assignments(features_normalized_test[:,3:],kmeans_centers)

print(test_cluster_assignments)

coded_data_test, coded_data_text_test = get_coded_observations(clustering_array_test)
coded_data_number_test = coded_data_test.astype(float)
# data = np.copy(clustering_array)
vector_quantized_data_test = np.hstack((coded_data_number_test[:,0:2],test_cluster_assignments[:,np.newaxis])).astype(np.int)
label_quantized_data_test = np.hstack((coded_data_text_test[:,0:2],test_cluster_assignments[:,np.newaxis]))

class_cnt_test = np.unique(label_quantized_data_test[:,0])
print(class_cnt_test)
ob_list_test =[]
label_list_test = []
for step_count,text_label in enumerate(class_cnt_test):
    class_cur = vector_quantized_data_test[vector_quantized_data_test[:,0]==step_count ]
    label_cur = label_quantized_data_test[label_quantized_data_test[:,0]==text_label ]
    
    # how many different observations there are in this unique class
    obs_in_class = len(np.unique(class_cur[:,1]))
    label_list_test.append((text_label+" ")*obs_in_class)
    
    for j in range(obs_in_class):
        ob    = class_cur[class_cur[:,1]==j][:,2]
        ob_list_test.append(ob)
label_list_test = (("".join(label_list_test)).strip()).split(" ")

def run_test_data(ob_list_test_par):
    for jth_ob in range(len(label_list_test)):
        lambda_list = []
        delta = 10**-100
        current_ob = np.copy(ob_list_test[jth_ob])
        for ith_ob in range(A_count):
            lambda_list.append(Lambda(As[ith_ob],Bs[ith_ob]+delta,PIs[ith_ob],label_list_train[ith_ob],ith_ob,ob_list[ith_ob],N_states,current_ob))
        print("Done Setting Estimation Parameters for Observation", jth_ob+1,"- Performing Prediction...")
        pred_list = []
        pred_list = [lambda_object.get_prediction() for lambda_object in lambda_list]
        print (pred_list)
        print ("The predicted label is", label_list_train[np.argmax(np.asarray(pred_list))]+".\n")

run_test_data(ob_list_test)
