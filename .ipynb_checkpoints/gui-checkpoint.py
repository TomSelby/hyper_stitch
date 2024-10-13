##mcg_const = 1
##mcg = 2
from tkinter import *
from tkinter import filedialog
from tkinter import StringVar
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import utils
import cv2
import sys
from sklearn.neighbors import NearestNeighbors
import torch
import kornia.feature as KF
from kornia_moons.viz import *
from kornia_moons.feature import *
from numba import njit
from numba import jit
from numba import prange
import time

        
class MainWindow():
    def __init__(self,main):
        self.main = main
        self.sift_judgement = ''
        self.main.title('Image Stitcher')
        self.main.geometry('1300x700')
        self.main.config(background = "white")
        self.stitch_mode = 'kornia'

        ## Menu Bar
        menubar = Menu(main)
        self.main.config(menu=menubar)
        self.file_menu = Menu(menubar)
        self.file_menu.add_command(label = 'Save transform list', command = self.save_transform_list)
        self.file_menu.add_command(label = 'Load transform list',command=self.load_transform_list)

        self.file_menu.add_command(label = 'Alter SIFT and RANSAC params', command = self.sift_ransac_params)
        self.file_menu.add_command(label = 'Change detection mode', command = self.change_detection)
        menubar.add_cascade(label = "File",menu=self.file_menu)
        self.last_filepath = '/'
        self.transform_dataframe = pd.DataFrame(columns = ['file1','file2','TM'])
        
        #Ransac and SIFT hyperparams
        self.ransac_recip_thresh= 0.99
        self.ransac_confidence = 0.99
        self.ransac_refine_iters = 10000
        self.numb_SIFT_points = 1000
        self.mode = 'stitching'
        
        
        ## Page
        self.transform = np.array([[1,0,0],[0,1,0]])
        
        tm_entry = StringVar()
        self.tm_entry_block = Entry(main,textvariable=tm_entry)
    
        self.button_load_tm = Button(main,
                                    text = "Perform TM from input",
                                    command = lambda: self.load_inputted_TM(tm_entry.get()))
        
        self.tm_from_selected_pts = Button(main,
                                    text = "Perform TM from\nselected points",
                                    command = self.tm_from_selection)
        
        self.button_explore_files = Button(main,
                                    text = "Browse Files",
                                    command = self.browseFiles)
        
        self.button_explore_save = Button(main,
                                    text = "Browse Files",
                                    command = self.browseSavePath)
        
        self.button_load_im1 = Button(main,
                                    text="Load First Image",
                                    command = lambda: self.loadimg(1))
        
        self.button_load_im2 = Button(main,
                                    text="Load Second Image",
                                    command = lambda: self.loadimg(2))
        
        self.button_sift = Button(main,
                                 text= "Detect keypoints and \nRANSAC estimation",
                                 command = self.siftestimation)
        
        self.button_icp = Button(main,
                                text = 'ICP refinement',
                                command = self.icp_refine)
        
        self.button_manual_points = Button(main,
                                    text = 'Select points manually',
                                    command = self.estimate_tm_manually)
        
        self.label_file_explorer = Label(main,
                            text = "Please select a file",
                            width = 70, height = 4,
                            fg = "blue")
        self.label_save_path = Label(main,
                                text = "Please select a project folder",
                                width=70, height =4,
                                fg = 'red')
        
        
        self.affine_label = Label(main,
                            text = str(self.transform),
                            width= 10, height=4,
                            fg = "black")
        
        self.button_save_transfom = Button(main, 
                                   text = 'Confirm Stitch and \nsave transform',
                                   command = self.add_transform_to_list)
 
        


        
        self.canvas1 = Canvas(main, height=500,width=500)
        self.canvas2 = Canvas(main, height=500,width=500)
        self.image_on_canvas1 = self.canvas1.create_image(0, 0, anchor='nw', image=None)
        self.image_on_canvas2 = self.canvas2.create_image(0, 0, anchor='nw', image=None)        
        self.label_file_explorer.grid(row = 0, column = 0,columnspan=1,sticky= N)
        self.button_explore_files.grid(row=1, column=0,columnspan = 1,sticky= N)
        self.button_explore_save.grid(row=1, column=1,columnspan = 1,sticky= N)

        self.label_save_path.grid(row =0,column=1,sticky=N)
        
        self.button_load_im1.grid(row = 2, column = 0, sticky = N)
        self.button_load_im2.grid(row = 2, column = 1, sticky = N)
        self.button_manual_points.grid(row =3, column = 2, sticky = N)
        self.canvas1.grid(row=3, column=0,columnspan = 1, sticky = N)
        self.canvas2.grid(row=3, column=1,columnspan = 1, sticky = N)
        self.affine_label.grid(row=3, column=2, sticky=W)
        self.button_save_transfom.grid(row=3, column=3, columnspan=1, sticky=W)
        self.button_sift.grid(row=4, column=0, columnspan=1, sticky=N)
        self.button_icp.grid(row=4, column=1, columnspan=1, sticky=N)
        self.tm_entry_block.grid(row=2, column=2)
        self.button_load_tm.grid(row=2, column=3,sticky = N)
        self.tm_from_selected_pts.grid(row=3, column=3, sticky = N)
       
        
    def korina_detection(self):
        def numpy_to_torch_tensor(npy):
            npy = npy - np.amin(npy)
            npy = 255*(npy/np.amax(npy))
            tensor = np.zeros((1,1,np.shape(npy)[0],np.shape(npy)[1]))
            tensor[0,0] = npy
            tensor = torch.FloatTensor(tensor)
            return tensor
        def get_matching_keypoints(lafs1, lafs2, idxs):
            mkpts1 = KF.get_laf_center(lafs1).squeeze()[idxs[:, 0]].detach().cpu().numpy()
            mkpts2 = KF.get_laf_center(lafs2).squeeze()[idxs[:, 1]].detach().cpu().numpy()
            return mkpts1, mkpts2
    

        device = torch.device("cpu")
        feature = KF.KeyNetAffNetHardNet(5000, True).eval().to(device)
        img1 = numpy_to_torch_tensor(self.imarray1)
        img0 = numpy_to_torch_tensor(self.imarray2)
        input_dict = {"image0":img0,
                     "image1":img1}
        feature = KF.KeyNetAffNetHardNet(5000, True).eval().to(device)
        adalam_config = {"device": device}
        hw1 = torch.tensor(img1.shape[2:])
        hw2 = torch.tensor(img1.shape[2:])
        with torch.inference_mode():
            lafs1, resps1, descs1 = feature(img0)
            lafs2, resps2, descs2 = feature(img1)
            dists, idxs = KF.match_adalam(
                descs1.squeeze(0),
                descs2.squeeze(0),
                lafs1,
                lafs2,  # Adalam takes into account also geometric information
                config=adalam_config,
                hw1=hw1,
                hw2=hw2,  # Adalam also benefits from knowing image size
            )
        print(f"{idxs.shape[0]} tentative matches with AdaLAM")
        mkpts1, mkpts2 = get_matching_keypoints(lafs1, lafs2, idxs)
        return mkpts1,mkpts2
        

    
    def change_detection(self):
        modes = np.array(['sift', 'brisk', 'combo','kornia']) # https://kornia-tutorials.readthedocs.io/en/latest/_nbs/image_matching_adalam.html
        current_mode = np.where(modes == self.stitch_mode)[0][0]
        
        if current_mode == 3:
            current_mode = -1
        self.stitch_mode = modes[current_mode+1]
        print(f'Detection mode changed to: {self.stitch_mode}')
        
    
       
        
    def tm_from_selection(self):
        self.transform,inliers = cv2.estimateAffinePartial2D(np.array(self.subplot1pts),np.array(self.subplot0pts),confidence= self.ransac_confidence,ransacReprojThreshold=self.ransac_recip_thresh,refineIters=self.ransac_refine_iters)
        
                
        self.affine_label.configure(self.affine_label, text=str(np.around(self.transform)))# Update affine label

        self.perform_transform()
         
            
    def estimate_tm_manually(self):
        
        def onclick(event):
    
            if event.dblclick:
                print(event.xdata,event.ydata)
                circle=plt.Circle((event.xdata,event.ydata),3,color='r')
                ax[event.inaxes.get_subplotspec().colspan[0]].add_patch(circle)
                fig.canvas.draw() #this line was missing earlier

                if event.inaxes.get_subplotspec().colspan[0] == 0:
                    self.subplot0pts.append([event.xdata,event.ydata])
                elif event.inaxes.get_subplotspec().colspan[0] == 1:
                    self.subplot1pts.append([event.xdata, event.ydata])


        self.subplot0pts = []
        self.subplot1pts = []


        fig,ax = plt.subplots(1,2)
        ax[0].imshow(self.npy1,cmap='gray')
        ax[1].imshow(self.npy2,cmap='gray')
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.suptitle('Select Correspondances with double click')

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        
    def browseFiles(self):
        self.filename = filedialog.askopenfilename(initialdir = self.last_filepath,
                                              title = "Select a File",
                                              filetypes = [("Image files",
                                                            "*.npy"),
                                                           ("All files",
                                                            "*.*")])
        self.label_file_explorer.configure(text="File: "+ self.filename)
        self.last_filepath = self.filename


    def browseSavePath(self):
        self.savepath = filedialog.askdirectory(initialdir = self.last_filepath,
                                              title = "Select a project folder")
        self.label_save_path.configure(text="Project folder: "+ self.savepath)
       
        self.last_filepath = self.savepath

    
        

    def loadimg(self,canvas_num):
        if canvas_num == 1:
            self.filename1 = self.filename
            self.npy1 = np.load(self.filename)
            npy = np.load(self.filename)
            npy = npy - np.amin(npy)
            npy = 255*(npy/np.amax(npy))
            self.imarray1 = npy.astype('uint8')
            img = ImageTk.PhotoImage(Image.fromarray(npy).resize((500, 500)))
            self.canvas1.image = img
            self.canvas1.itemconfig(self.image_on_canvas1, image=img)
            self.canvas1_filename = self.filename
            
        if canvas_num == 2:
            self.filename2 = self.filename
            self.npy2 = np.load(self.filename)
            npy = np.load(self.filename)
            npy = npy - np.amin(npy)
            npy = 255*(npy/np.amax(npy))
            self.imarray2 = npy.astype('uint8')
            img = ImageTk.PhotoImage(Image.fromarray(npy).resize((500, 500)))
            self.canvas2.image = img
            self.canvas2.itemconfig(self.image_on_canvas2, image=img)
            self.canvas2_filename = self.filename
            
            
            
    def siftestimation(self):
        
        sift1 = cv2.SIFT_create(self.numb_SIFT_points)
        sift2 = cv2.SIFT_create(self.numb_SIFT_points)
        orb1 = cv2.ORB_create(self.numb_SIFT_points)   
        orb2 = cv2.ORB_create(self.numb_SIFT_points)
        brisk1 = cv2.BRISK_create(self.numb_SIFT_points)
        brisk2 = cv2.BRISK_create(self.numb_SIFT_points)
     

        
        self.features1b, self.descriptors1b = brisk1.detectAndCompute(self.imarray1,None)
        self.features2b, self.descriptors2b = brisk2.detectAndCompute(self.imarray2,None)    

        self.features1s, self.descriptors1s = sift1.detectAndCompute(self.imarray1, None)
        self.features2s, self.descriptors2s = sift2.detectAndCompute(self.imarray2, None)
        
      
        bf_b = cv2.BFMatcher(crossCheck = True)
        bf_s = cv2.BFMatcher(crossCheck = True)
        
        matches_b = bf_b.match(self.descriptors2b,self.descriptors1b)
        matches_s = bf_s.match(self.descriptors2s,self.descriptors1s)
       


        ## Get the points of xy1 and xy2 for the best 4 points ** NOTE- as images are the same mag but differet dimensions this makes things difficult
        self.src_points = []
        self.des_points = []
        
        
        
        if self.stitch_mode == 'sift':
            for match in matches_s:
                    feature1, feature2 = (self.features2s[match.queryIdx]), (self.features1s[match.trainIdx])
                    src_point  = int(feature1.pt[0]), int(feature1.pt[1])
                    des_point  = int(feature2.pt[0]),int(feature2.pt[1])
                    self.src_points.append(src_point)
                    self.des_points.append(des_point)
        
        elif self.stitch_mode == 'brisk':
            for match in matches_b:
                feature1, feature2 = (self.features2b[match.queryIdx]), (self.features1b[match.trainIdx])
                src_point  = int(feature1.pt[0]), int(feature1.pt[1])
                des_point  = int(feature2.pt[0]),int(feature2.pt[1])
                self.src_points.append(src_point)
                self.des_points.append(des_point)
        

        elif self.stitch_mode == 'kornia':
            self.src_points,self.des_points = self.korina_detection()
            
        elif self.stitch_mode == 'combo':
            matches_list = [matches_b, matches_s]
            features_lits = [(self.features2b, self.features1b),(self.features2s,self.features1s)]
            for i in range(2):# Can get rid of this for loop if only using one of SIFT/orb/brisk
                matches = matches_list[i]
                self.features2, self.features1 = features_lits[i]
                for match in matches:
                    feature1, feature2 = (self.features2[match.queryIdx]), (self.features1[match.trainIdx])
                    src_point  = int(feature1.pt[0]), int(feature1.pt[1])
                    des_point  = int(feature2.pt[0]),int(feature2.pt[1])
                    self.src_points.append(src_point)
                    self.des_points.append(des_point)
                
            src_points,des_points = self.korina_detection()
            self.src_points.append(src_point)
            self.des_points.append(des_point)       
        
        try:
            self.transform,inliers = cv2.estimateAffinePartial2D(np.array(self.src_points), np.array(self.des_points),confidence= self.ransac_confidence,ransacReprojThreshold=self.ransac_recip_thresh,refineIters=self.ransac_refine_iters)
            print(f"{inliers.sum()} inliers")
            self.affine_label.configure(self.affine_label, text=str(np.around(self.transform)))# Update affine label
            print(self.transform)
            
          
            #Add a santity check
            if self.mode == 'stitching':
                if (self.transform[0][0] < 0.95) & (self.transform[0][0] > 1.05):
                    print(self.transform, self.transform[0][0])
                    return
                self.perform_transform() #Call the transform function
       
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(e)
            print(exc_type, exc_tb.tb_lineno)

    
            
    def sift_ransac_params(self):
        print('Current Pararmaters:')
        
        print(f'RANSAC Recip Threshold: {self.ransac_recip_thresh}\nRANSAC Confidence: {self.ransac_confidence}\nRANSAC Iterations: {self.ransac_refine_iters}\nNumber of SIFT keypoints: {self.numb_SIFT_points}')
        print('##############################')

        self.ransac_recip_thresh = float(input('Input a new RANAC recip thresh: '))
        self.ransac_confidence = float(input('Input a new RANSAC confidence: '))
        self.ransac_refine_iters = int(input('Input a new max number of RANSAC iterations: '))
        self.numb_SIFT_points = int(input('Input a new number of SIFT keypoitns: '))
        
        print('##############################')
        print('New Parameters:')
        print(f'RANSAC Recip Threshold: {self.ransac_recip_thresh}\nRANSAC Confidence: {self.ransac_confidence}\nRANSAC Iterations: {self.ransac_refine_iters}\nNumber of SIFT keypoints: {self.numb_SIFT_points}')

    
      
    def icp_refine(self):
        def xyz_from_arr(arr):
            xyz = np.zeros((np.shape(arr)[0]*np.shape(arr)[1], 3))
            for i in range(np.shape(arr)[1]):
                for j in range(np.shape(arr)[0]):
                    xyz[i*np.shape(arr)[0]+j] = j,i,arr[j,i] #0 and 1 jand i
            return xyz
        
        @jit(nopython=True,parallel=True)
        def nearest_neighbor(src, dst):
            # dist = np.linalg.norm(src[:, None] - dst, axis=-1)
            indicies = np.zeros(len(src))
            for i in prange(len(src)):
                src_point = src[i]
                distances = np.sqrt(((src_point[0]-dst[:,0])**2)+((src_point[1]-dst[:,1]))**2+((src_point[2]-dst[:,2])**2))
                indicies[i] = np.argmin(distances)
            return indicies
            
         


        def icp(self):
            
            where_overlap = np.where((self.t_canvas1>0) & (self.t_canvas2 >0))
            min_x,max_x = np.amin(where_overlap[0]), np.amax(where_overlap[0])
            min_y,max_y = np.amin(where_overlap[1]), np.amax(where_overlap[1])

            src = self.t_canvas1[min_x:max_x, min_y:max_y]
            dest = self.t_canvas2[min_x:max_x, min_y:max_y]
            
            xyz0 = xyz_from_arr(src)
            xyz1 = xyz_from_arr(dest)
            
            ind = nearest_neighbor(xyz0, xyz1)
       
            sum_change_x = 0
            sum_change_y = 0

            for i,point in enumerate(ind):
                change = xyz0[i]-xyz1[int(point)]
                sum_change_y += change[0]#swapped
                sum_change_x += change[1]
    
        
            av_change_x = sum_change_x/len(ind)
            av_change_y = sum_change_y/len(ind)
            print(f'X change:{av_change_x}')
            print(f'Y change:{av_change_y}')
            self.transform[0,2] += av_change_x # - and plus swapped
            self.transform[1,2] += av_change_y
            self.perform_transform(plot=False)
            
            
        self.mode = 'icp'
        t0 = time.time()
        for i in range(10):
            
            icp(self)
            print(f'Iteration: {i}, Time: {time.time()-t0} s')
        print('ICP complete')
        self.mode = 'stitching'
        self.perform_transform()

        
        
        
        
    ## Functions from file dropdown    
    def add_transform_to_list(self):
        df2 = pd.DataFrame([[self.canvas1_filename,self.canvas2_filename,self.transform]],columns = ['file1','file2', 'TM'])
        self.transform_dataframe = pd.concat([self.transform_dataframe,df2])
            
        if self.mode == 'stitching':    
            os.remove(self.canvas1_filename)
    
        
        np.save(self.canvas2_filename,self.result)
        
        
        ######### CHANGE THIS #########
        os.chdir(self.savepath)
        ######### CHANGE THIS #########


        
        self.save_transform_list()
        os.chdir(r'C:\Users\tas72\Documents\GitHub\Hyper_stitch')
        print('transform_added')
        
    def save_transform_list(self):
        print('Transform df head:')
        print(self.transform_dataframe.head())
        print('Transform df last item:')
        print(self.transform_dataframe.iloc[-1])
        self.transform_dataframe.to_pickle('transform_list')
        print('Transform list saved at:')
        print(os.getcwd())
        
    def load_transform_list(self):
        path = filedialog.askopenfilename(initialdir = self.last_filepath,
                                              title = "Select a File",
                                              filetypes = (("Pickle Files",
                                                            ""),
                                                           ("all files",
                                                            "*.*")))
        
        self.last_filepath = path
        
        self.transform_dataframe = pd.read_pickle(path)
        print('Loaded')
        print(self.transform_dataframe.head())
        print(f'Total length: {len(self.transform_dataframe)}')
                           
                
                

        
    def load_inputted_TM(self,inputted):
        inputted = list(map(int, inputted.split(',')))
        tm = np.array([[inputted[0],inputted[1],inputted[2]],
                        [inputted[3],inputted[-2],inputted[-1]]])
        self.transform = tm # Save new transform
        self.affine_label.configure(self.affine_label, text=str(tm))# Update affine label
        self.perform_transform() #Call the transform function
        
        
        
    def perform_transform(self,plot= True):
        print(self.transform)
        scale_factor = max(self.transform[0][0], self.transform[1][1])
        mcg_const_shape = np.shape(self.imarray1)
        mcg_shape = np.shape(self.imarray2)
        self.t_canvas1 = np.zeros((mcg_const_shape[0]+int((2*mcg_shape[0]*scale_factor))+500,mcg_const_shape[1]+int((2*mcg_shape[1]*scale_factor))+500)) # Want to be able to fit 2* image around your const img
        canvas2 = np.zeros((mcg_const_shape[0]+int((2*mcg_shape[0]*scale_factor))+500,mcg_const_shape[1]+int((2*mcg_shape[1]*scale_factor))+500))
        canvas_shape = np.shape(canvas2)
        print(canvas_shape)
        
        top_left_corn = (int((canvas_shape[0]-mcg_const_shape[0])/2),int((canvas_shape[1]-mcg_const_shape[1])/2))
        self.t_canvas1[top_left_corn[0]:top_left_corn[0] + int(mcg_const_shape[0]),top_left_corn[1]:top_left_corn[1] + int(mcg_const_shape[1])] = self.imarray1
        canvas2[top_left_corn[0]:top_left_corn[0] + int(mcg_shape[0]),top_left_corn[1]:top_left_corn[1] + int(mcg_shape[1]) ] = self.imarray2

     
        M = np.float32([[1, 0,-top_left_corn[1]+(top_left_corn[1]/self.transform[1][1])], #this correction of the mcg image was placed in the center
                         [0, 1,-top_left_corn[0]+(top_left_corn[0]/self.transform[0][0])]])

        self.t_canvas2 = cv2.warpAffine(canvas2, M, (canvas_shape[1],canvas_shape[0]))
        
        self.t_canvas2 = cv2.warpAffine(self.t_canvas2, np.float32(self.transform), (canvas_shape[1],canvas_shape[0]))      
 
        self.result = cv2.addWeighted(self.t_canvas1, 1,self.t_canvas2, 1 , 0.0)
        if plot ==True:
            self.plot_result()
        ## Get where result != canvas1 or canvas2 (the overlap) and make it = canvas1 (could equally use canvas2) so as not to get bright spot when during image blend
        not_equal_to_canv1 = self.result != self.t_canvas1
        not_equal_to_canv2 = self.result != self.t_canvas2
        self.result = np.where(np.logical_and(not_equal_to_canv1, not_equal_to_canv2),self.t_canvas1,self.result)      
        ## Cut and plot
        self.result = self.result[~np.all(self.result == 0, axis=1)]
        self.result = self.result[:,~(self.result==0).all(0)]
        if plot == True:
            self.plot_result()
        

    def plot_result(self):
        ## Create a pop up with result
        
        fig,ax = plt.subplots()
        ax.imshow(self.result,cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
        
# Create the root window
root = Tk()
MainWindow(root)
# Let the window wait for any events
root.mainloop()