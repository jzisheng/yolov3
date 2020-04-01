import aug_util as aug 
import wv_util as wv
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import csv
import tqdm
import pickle
from sklearn.model_selection import train_test_split
import itertools
import cv2 as cv
import glob
import random
import os
random.seed(1)

all_classes =  \
"""11:Fixed-wing Aircraft
12:Small Aircraft
13:Cargo Plane
15:Helicopter
17:Passenger Vehicle
18:Small Car
19:Bus
20:Pickup Truck
21:Utility Truck
23:Truck
24:Cargo Truck
25:Truck w/Box
26:Truck Tractor
27:Trailer
28:Truck w/Flatbed
29:Truck w/Liquid
32:Crane Truck
33:Railway Vehicle
34:Passenger Car
35:Cargo Car
36:Flat Car
37:Tank car
38:Locomotive
40:Maritime Vessel
41:Motorboat
42:Sailboat
44:Tugboat
45:Barge
47:Fishing Vessel
49:Ferry
50:Yacht
51:Container Ship
52:Oil Tanker
53:Engineering Vehicle
54:Tower crane
55:Container Crane
56:Reach Stacker
57:Straddle Carrier
59:Mobile Crane
60:Dump Truck
61:Haul Truck
62:Scraper/Tractor
63:Front loader/Bulldozer
64:Excavator
65:Cement Mixer
66:Ground Grader
71:Hut/Tent
72:Shed
73:Building
74:Aircraft Hangar
76:Damaged Building
77:Facility
79:Construction Site
83:Vehicle Lot
84:Helipad
86:Storage Tank
89:Shipping container lot
91:Shipping Container
93:Pylon
94:Tower"""
all_labels_dict = {}
all_labels_dict_int = {}
for line in all_classes.split("\n"):
    key,label = line.split(":")
    all_labels_dict[key]=label
    
    

class XviewDataset():
    def __init__(self,grouped_classes_,labels_,coords_,chips_,classes_):
        self.grouped_classes = grouped_classes_
        self.all_classes = list(itertools.chain.from_iterable(grouped_classes_))
        
        self.labels = labels_
        self.chips = chips_        
        self.coords = coords_
        self.classes = classes_
        pass
    

    def getGroupedClassIdx(self,gcs,classLabel):
        for idx,labels in enumerate(gcs):
            if classLabel in labels:
                return idx

    def getGroupedTransformDict(self):
        transDict = {}
        for idx,grouped_class in enumerate(self.grouped_classes):
            for label in (grouped_class):
                #print("{} : {}".format(idx,label))
                transDict[label] = idx
                pass
            pass
        return (transDict)
        
    def getAllGroupTransformDict(self):
        #print("---")
        transDict = {}
        for idx,label in enumerate(self.all_classes):
            #print("{} : {}".format(idx,label))
            transDict[label] = idx
        return (transDict)
                
    def getLabelCounts(self):
        """ 
        given a chips and classes, returns dataframe of 
        new label groupings counts
        """
        chip_names = np.unique(self.chips)
        gcResults = np.zeros((len(chip_names),len(self.grouped_classes)))
        allGcResults = np.zeros((len(chip_names),len(self.all_classes)))
        
        gcTransDict = self.getGroupedTransformDict()
        allGcTransDict = self.getAllGroupTransformDict()
        
        chip_strs = []
        for c_idx, c in tqdm.tqdm(enumerate(chip_names)):
            chip_strs.append(c)
            # Get all classes in this chip
            classes_chip = self.classes[self.chips==c]
            # Filter to only get selected classes
            mask = np.isin(classes_chip,self.all_classes)
            classes_chip = classes_chip[mask]
            # Now group and count classes by new indexes
            self.classesChipsCount(gcResults,classes_chip,gcTransDict,c_idx)
            self.classesChipsCount(allGcResults,classes_chip,allGcTransDict,c_idx)
            pass
        chip_strs = np.array(chip_strs).reshape(-1,1)
        return np.hstack((chip_strs,gcResults)),np.hstack((chip_strs,allGcResults))

    def checkThreshold(self,distr1, distr2, thres):
        if (len(distr1) != len(distr2)):
            print("columns' numbers don't fit.")
            return -1
        for i in range(len(distr1)):
            diff = abs(distr1[i] - distr2[i])
            if diff > thres:
                return False
        return True
    def classesChipsCount(self,results,classes_chip,transDict,c_idx):
        # Get counts for grouped classes
        gcLabels = (np.array([transDict[x] for x in classes_chip]))
        labels,counts = np.unique(gcLabels,return_counts=True)
        # Put counts for tif in grouped classes
        np.put(results[c_idx,:],labels.astype(np.int64),np.array(counts,dtype=np.int64))
        pass
    
    def getDistribution(self,data, selected_indexes,getNum=False):
        #print(data.astype(float))
        res = (np.sum(np.array(data[selected_indexes,:].astype(float)),axis=0))
        if(getNum): 
            dividend = 1
        else: 
            dividend = np.sum(res,axis=0)
        return (res/dividend)
    
    def findBalance(self,data, train_percent, thres,debug=False):
        class_num = len(data[0])
        for i in range(1000000):
            tr_set, te_set = train_test_split(np.array(list(range(len(data)))),\
                                              test_size=1-train_percent)
            tr_d = self.getDistribution(data, tr_set)
            te_d = self.getDistribution(data, te_set)
            
            tr_numd = self.getDistribution(data, tr_set,getNum=True)
            te_numd = self.getDistribution(data, te_set,getNum=True)
            check = self.checkThreshold(tr_d, te_d, thres)
            if (check == True) or debug:
                return (tr_set, te_set),(tr_numd,te_numd)
            elif (check == -1):
                return -1

        return [], []
    def indToTifName(self,data, inds):
        res = []
        for ind in inds:
            res.append(data[ind][0])
        return res
    
    def getClassList(self):
        result = []
        for idx,label in enumerate(self.all_classes):
            temp = all_labels_dict[str(label)] 
            result.append(temp)
        return result
    
    def getGroupedClassList(self):
        result = []
        for idx,labels in enumerate(self.grouped_classes):
            temp = [self.labels[idx] for label in labels]
            result.extend(temp)
        return result
    
    
    def getResultsDistribution(self,labelCounts,tifNames):
        mask = np.isin(labelCounts[:,0],tifNames)
        result = ( np.array(labelCounts[mask,1:],dtype=float) )
        result = np.sum(result,axis=0).reshape(-1,1)
        np_agc = (np.array(self.getGroupedClassList()).reshape(-1,1))
        np_gcIdx = np.array(self.getClassList()).reshape(-1,1)
        df =  pd.DataFrame(np.hstack( (np_agc,np_gcIdx , result)))
        df.columns = ["grouped cls","og label",'count']
        return df
    
    def splitTrainTest(self,debug=False):
        allGroupedClasses = []
        for a in self.grouped_classes:
            allGroupedClasses.extend(a)
        #allLabelCounts = self.getLabelCounts()
        labelCounts,allLabelCounts = self.getLabelCounts()
        idxs,numd = self.findBalance(labelCounts[:,1:], 0.8, 0.1,debug=debug)
        train_ind, test_ind = idxs
        
        train_tifs = self.indToTifName(labelCounts,train_ind)
        test_tifs = self.indToTifName(labelCounts, test_ind)
        
        mask = (np.isin(allLabelCounts[:,0],train_tifs))
        
        df_train = (self.getResultsDistribution(allLabelCounts,train_tifs))
        df_test = (self.getResultsDistribution(allLabelCounts,test_tifs))

        gc_df_train = (pd.DataFrame(np.hstack(( np.array(self.labels).reshape(-1,1),
                        numd[0].reshape(-1,1) )) ))
        gc_df_test = (pd.DataFrame(np.hstack(( np.array(self.labels).reshape(-1,1),
                        numd[1].reshape(-1,1) )) ))
        return (train_tifs,test_tifs),((df_train,df_test),(gc_df_train,gc_df_test))
      


class DarkNetFormatter():
    def __init__(self,output_dir_,input_dir_,coords_,chips_,classes_,grouped_classes_):
        self.output_dir = output_dir_
        self.input_dir = input_dir_
        self.chips = chips_        
        self.coords = coords_
        self.classes = classes_
        self.grouped_classes = grouped_classes_
        self.res_native=30
        self.all_c_ids = []
        pass
    
    def filterClasses(self,chip_coords,chip_classes,grouped_classes):
        filtered_classes = list(itertools.chain.from_iterable(self.grouped_classes))
        mask = (np.isin(chip_classes,filtered_classes))
        chip_coords, chip_classes = chip_coords[mask], chip_classes[mask]

        for idx, g_cls in enumerate(self.grouped_classes):

            mask = (np.isin(chip_classes,g_cls))
            chip_classes[mask] = idx
        return chip_coords,chip_classes
        pass

    def plotDarknetFmt(self,c_img,x_center,y_center,ws,hs,c_cls,szx,szy):
        fig,ax = plt.subplots(1,figsize=(5,5))
        ax.imshow(c_img)
        for didx in range(c_cls.shape[0]):
            x,y = x_center[didx]*szx,y_center[didx]*szy
            w,h = ws[didx]*szx,hs[didx]*szy
            x1,y1 = x-(w/2), y-(h/2)
            w1,h1 = w,h
            rect = patches.Rectangle((x1,y1),w1,h1,\
                                     linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            break
        plt.show()
        pass

    def toDarknetFmt(self,c_box,c_cls,c_img,c_ids, debug=False):
        szx,szy,_ = c_img.shape
        c_box[:,0],c_box[:,2] = c_box[:,0]/szx,c_box[:,2]/szx
        c_box[:,1],c_box[:,3] = c_box[:,1]/szy,c_box[:,3]/szy
        xmin,ymin,xmax,ymax = c_box[:,0],c_box[:,1],c_box[:,2],c_box[:,3]
        ws,hs = (xmax-xmin), (ymax-ymin)
        x_center, y_center = xmin+(ws/2),ymin+(hs/2)
        # Visualize using mpl
        if debug == True:
            plotDarknetFmt(c_img,x_center,y_center,ws,hs,c_cls,szx,szy)
            pass
        # result = np.vstack((c_cls,x_center,y_center,ws,hs)).T            
        result = np.vstack((c_cls,x_center,y_center,ws,hs,c_ids)).T
        result[:,1:5] = np.clip(result[:,1:5],0.001,0.999)
        assert (result[:, 1:5] > 0).all(), "values less than 0"
        assert (result[:, 1:5] <= 1).all(),"Values not normalized"
        return result

    def checkDir(self,filepath):
        """ passed a filepath string, checks if it dne
        if it does not exists makes directory"""
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
    def downsampleImage(self,img,res_native=30,res_out=120):
        kernel_sz = int(0.5*(res_out/res_native))
        scale_percent = res_native/res_out
        width = int(img.shape[1] * scale_percent )
        height = int(img.shape[0] * scale_percent)
        dim = (width, height)
        sigma = 0.3*((kernel_sz-1)*0.5 - 1) + 0.8
        blur = cv.blur(img,(kernel_sz,kernel_sz),borderType=cv.BORDER_REFLECT)
        resized = cv.resize(blur, dim, interpolation = cv.INTER_AREA)
        return resized
    
    def parseChip(self,c_img, c_box, c_cls,c_ids,
                  img_num,c_dir,res_out=30,showImg = False):
        # Parses chips, saves chip image, and also saves corresponding labels
        fnames = []
        
        outputImgDir = "{}labels/".format(c_dir)
        outputLabelDir = "{}images/".format(c_dir)
        self.checkDir(outputImgDir)
        self.checkDir(outputLabelDir)
        
        images = []
        sboxes,sclasses,simgs = [],[],[]
        
        for c_idx in range(c_img.shape[0]):
            c_name = "{:06}_{:02}".format(int(img_num), c_idx)
            sbox,scls,simg,sid = \
                c_box[c_idx],c_cls[c_idx],c_img[c_idx], c_ids[c_idx]
            #print(sid)
            if showImg:
                labelled = aug.draw_bboxes(simg,sbox)
                plt.figure(figsize=(10,10))
                plt.imshow(labelled)
                #print(c_box[:,4])
                plt.title(str(c_name)+" "+str(simg.shape))
                plt.axis('off')
                break
                
            # Change chip into darknet format, and save
            result = self.toDarknetFmt(sbox,scls,simg,sid)
            ff_l = "{}labels/{}.txt".format(c_dir,c_name)
            np.savetxt(ff_l, result, fmt='%i %1.6f %1.6f %1.6f %1.6f %i')
            # Save image to specified dir
            ff_i = "{}images/{}.jpg".format(c_dir,c_name)
            
            Image.fromarray(simg).save(ff_i)
            # Append file name to list
            fnames.append("{}images/{}.jpg".format(c_dir,c_name))
            pass
        return fnames
    
    def exportChipImages(self,image_paths,c_dir,set_str,res_out=30,
                         showImg=False,shape=(600,600),chipImage=False):
        fnames = []
        #image_paths = sorted(image_paths)
        for img_pth in image_paths:
            try:
                img_pth = self.input_dir+'train_images/'+img_pth
                img_name = img_pth.split("/")[-1]
                img_num = img_name.split(".")[0]
                arr = wv.get_image(img_pth)
                chip_coords = self.coords[self.chips==img_name]
                chip_classes = self.classes[self.chips==img_name].astype(np.int64)
                chip_coords,chip_classes = \
                    self.filterClasses(chip_coords,chip_classes,self.grouped_classes)
                # Code for downsampling the image
                scale = self.res_native/res_out
                if res_out != 30:
                    arr = self.downsampleImage(arr,res_out=res_out)
                    chip_coords[:,:4] = chip_coords[:,:4]*scale
                # Chip the tif image into tiles
                c_img, c_box, c_cls,c_ids  = chip_image(img=arr, coords=chip_coords, 
                                                           classes=chip_classes, shape=shape,
                                                           chipImage=chipImage)
                if showImg:
                    result = []
                    for key in c_cls.keys():
                        result.extend(c_cls[key])
                    print("number of classes: {}".format(len(result)))
                    for i,img in enumerate(c_img[:5]):
                        labelled = aug.draw_bboxes(c_img[i],c_box[i])
                        plt.imshow(labelled)
                        plt.axis('off')
                        plt.show()
                        pass
                    break
                    pass
                c_fnames = self.parseChip(c_img, c_box, c_cls, c_ids,
                                          img_num, c_dir,res_out)
                fnames.extend(c_fnames)

            except FileNotFoundError as e:
                print(e)
                pass
            pass
        
        lines = sorted(fnames)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            pass
        
        outputTxtPath = self.output_dir+"xview_img_{}.txt".format(set_str)

        print(outputTxtPath)
        if os.path.exists(outputTxtPath):
            os.remove(outputTxtPath)
            pass
        
        with open(outputTxtPath, mode='w', encoding='utf-8') as myfile:
            myfile.write('\n'.join(lines))
        pass

    def transformDatum(self,datum,res_out=30,shape=(416,416),
                       chipImage=False,showImg=False):
        """
        takes in as input list of tuples corresponding to
        first string of subset, and training file indexes
        ("train", [training file idxs])
        """
        for (data_str, data_files) in datum:
            self.exportChipImages(data_files,self.output_dir,data_str,
                                  res_out=res_out,shape=shape,showImg=showImg,
                                  chipImage=chipImage)
            pass
        pass
    
import math
def chip_image(img,coords,classes,shape=(300,300),chipImage=False):
    """
    Chip an image and get relative coordinates and classes.  Bounding boxes that pass into
        multiple chips are clipped: each portion that is in a chip is labeled. For example,
        half a building will be labeled if it is cut off in a chip. If there are no boxes,
        the boxes array will be [[0,0,0,0]] and classes [0].
        Note: This chip_image method is only tested on xView data-- 
        there are some image manipulations that can mess up different images.

    Args:
        img: the image to be chipped in array format
        coords: an (N,5) array of bounding box coordinates with the last representing ids
        classes: an (N,1) array of classes for each bounding box
        shape: an (W,H) tuple indicating width and height of chips

    Output:
        An image array of shape (M,W,H,C), where M is the number of chips,
        W and H are the dimensions of the image, and C is the number of color
        channels.  Also returns boxes and classes dictionaries for each corresponding chip.
    """
    assert coords.shape[1]==5 ,"Invalid dim: No unique coord idx labels to the 5th col"
    
    height,width,_ = img.shape
    if chipImage == False:
        # don't chip image and return original size
        wn,hn = np.max(img.shape),np.max(img.shape)        
        w_num,h_num = 1,1
    else:
        height,width,_ = img.shape
        wn,hn = shape
        w_num,h_num = (int(math.ceil(width/wn)),int(math.ceil(height/hn)))
        pass
    
    w_new,h_new = wn*(w_num),hn*(h_num)
    # Make new image with padded edges
    temp_img = np.zeros((h_new,w_new,3))
    temp_img[:height,:width,:] = img
    img = temp_img
    # Resulting images
    images = np.zeros((w_num*h_num,hn,wn,3))
    
    total_boxes = {}
    total_classes = {}
    total_ids = {}
    
    k = 0
    for i in range(w_num):
        for j in range(h_num):
            x = np.logical_or( np.logical_and((coords[:,0]<((i+1)*wn)),(coords[:,0]>(i*wn))),
                               np.logical_and((coords[:,2]<((i+1)*wn)),(coords[:,2]>(i*wn))))
            out = coords[x]
            y = np.logical_or( np.logical_and((out[:,1]<((j+1)*hn)),(out[:,1]>(j*hn))),
                               np.logical_and((out[:,3]<((j+1)*hn)),(out[:,3]>(j*hn))))
            outn = out[y]
            out = np.transpose(np.vstack((np.clip(outn[:,0]-(wn*i),0,wn),
                                          np.clip(outn[:,1]-(hn*j),0,hn),
                                          np.clip(outn[:,2]-(wn*i),0,wn),
                                          np.clip(outn[:,3]-(hn*j),0,hn))))
            box_classes = classes[x][y]
            bbox_ids = coords[x][y][:,4]
            
            if out.shape[0] != 0:
                total_boxes[k] = out
                total_classes[k] = box_classes
                total_ids[k] = bbox_ids
            else:
                total_boxes[k] = np.array([[0,0,0,0]])
                total_classes[k] = np.array([0])
                total_ids[k] = np.array([0])                
            
            chip = img[hn*j:hn*(j+1),wn*i:wn*(i+1),:3]
            images[k]=chip
            
            k = k + 1
            pass
    
    return images.astype(np.uint8),total_boxes,total_classes,total_ids

#if __name__ == "main":
# Load dataset
'''
coords1, chips1, classes1 = wv.get_labels('/data/zjc4//xView_train.geojson')
# Input desired classes
grouped_classes = [[77,73],[11,12],[13],[17,18,20,21],
       [19,23,24,25,28,29,60,61,65,26],[41,42,50,40,44,45,47,49]]
labels = ["building and facility" ,"small aircraft", 
          "large aircraft","vehicles","bus","boat"]
#print(labels[0])
#showClassExample(grouped_classes[0],chip_name = "1694.tif")
print(len(chips1))
idxs = range(0,300000)
xdataset = XviewDataset(grouped_classes,labels, coords1[idxs],chips1[idxs],classes1[idxs])
data_sets = xdataset.splitTrainValidTest(chips1[idxs],classes1[idxs])
string_sets = ["train","valid","test"]
dnf = DarkNetFormatter(output_dir_ = "/data/zjc4/chipped/data/",
                       input_dir_="/data/zjc4/",
                       coords_ = coords1[idxs],
                       chips_ = chips1[idxs],
                       classes_ = classes1[idxs],
                       grouped_classes_=grouped_classes)
datum = list(zip(string_sets,data_sets))
dnf.transformDatum(datum)
'''
