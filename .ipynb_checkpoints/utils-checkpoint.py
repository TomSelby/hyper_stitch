import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm



def des_from_scr(scr_points,scr_shape,TM):
    des_points = np.zeros_like(scr_points)
    for i in range(scr_shape[0]):
        for j in range(scr_shape[1]):
            des_points[i,j] = np.dot(TM[:2,:2],scr_points[i,j])+TM[:,2]
    return des_points, np.shape(des_points)
   
def zeros_cut(result):
    y_length = len(np.all(result == 0, axis=1)) # y-axis cut
    half_y = y_length//2 #splt in two
    y_halfed = np.all(result == 0, axis=1)[:half_y] 
    y_trans = len(y_halfed[y_halfed==True])# Only want values that are in top left corner and True
    
    x_length = len((result==0).all(0)) #x axis cut 
    half_x = x_length//2 
    x_halfed=((result==0).all(0))[:half_x]
    x_trans = len(x_halfed[x_halfed ==True])
    
    
    return [x_trans,y_trans]
# class points_dataframe():
#     def __init__(self,savepath):
#         self.savepath = savepath
#         print(self.savepath)
#         self.init_points_df()
def init_points_df(savepath):
    os.chdir(savepath) # Change directory to project folder
    points_df = pd.DataFrame(columns = ['original_filename','tracking_filename','des_points'])
    files = glob.glob('*')
    print(files)
    points_df['original_filename'] = files
    points_df['tracking_filename'] = files
    #set up with start points
    for k in tqdm(range(len(files))):
        file = np.load(files[k],allow_pickle=True)
        file_shape = np.shape(file)
        points_df['des_points'][k] =  np.shape(file)
        start_points = np.zeros((file_shape[0],file_shape[1],2))# 1/0
        for j in range(file_shape[0]):
            for i in range(file_shape[1]):
                start_points[i,j] = (i,j) #j/i
        points_df['des_points'][k] = start_points
        points_df.to_pickle('points_df.pkl')
    
    #def update_points_df(self):
        
 

    
        
            