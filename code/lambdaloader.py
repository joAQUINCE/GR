import os
import numpy as np

import random

list_par = []
os.chdir(os.path.dirname(os.path.realpath(__file__)))




def load_data(folder_par):


    file_count = len([name for name in os.listdir(folder_par)])
    random_file = random.choice(os.listdir(folder_par))
    matrix_dim = np.loadtxt(os.path.join(folder_par,random_file)).shape
    
    object_shape = ", ".join([str(file_count),", ".join([str(i) for i in matrix_dim])])
    object_shape = object_shape.split(",")
    object_shape = np.array([int(i) for i in object_shape])
    #print(object_shape)
    object_array = np.empty(object_shape)
    
    
    for count in range(file_count):
    
        object_array[count] = np.loadtxt(os.path.join(folder_par,str(count)+".txt"))
        # matrix_dim = ([np.loadtxt((os.path.join(folder_par,name))).shape for name in os.listdir(folder_par)])
        # list_par.append(np.loadtxt(os.path.join(folder_par,filename)))
        
    return object_array, file_count


#(os.path.join(folder_par,filename)