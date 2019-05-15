#=========================================================================
#   Brain Corp. Code Sample: 2019 Summer Intern - R&D Machine Learning
#   Author: Brad Magnetta
#-------------------------------------------------------------------------
#   Goals:      - Train and test object detection to identify the center
#                   position of cell phones in images
#
#   Givens:     - we know each image only has a single cell phone
#               - we only need to detect one type of cell phone
#
#   Validation: - image_detection folder holds the annotated images of our
#                   model's detected objects
#               - error and current model parameters are stored in mp.pkl
#=========================================================================

import check_dep
check_dep.check()
import sys
import math
import os
from random import randint
import pickle
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#ignores AVX2 message
tf.logging.set_verbosity(tf.logging.FATAL)#ignores dep. message
from imageai.Detection import ObjectDetection
from PIL import Image
import train    #keras-yolo3

#=========================================================================

class Phone:
    
    def __init__(self, image, xmid, ymid, folder):
        self.folder = folder
        self.image = image#type string
        self.xmid = xmid
        self.ymid = ymid
        self.xmid_pred = -1.
        self.ymid_pred = -1.
        self.error = bool()

    def set_prediction(self, box):
        #-------------------------------------------
        #   Objective:  scale predicted location of
        #               mid point of phone
        #-------------------------------------------
        im = Image.open(os.path.join(self.folder,self.image))
        xmax,ymax=im.size[0],im.size[1]
        im.close()
        self.xmid_pred=(float(box[2]-box[0])/2.+float(box[0]))/float(xmax)
        self.ymid_pred=(float(box[3]-box[1])/2.+float(box[1]))/float(ymax)

    def get_prediction(self, detector, min_prob):
        #-------------------------------------------
        #   Objective:  get predicted location of
        #               mid point of phone
        #   ----------------------------------------
        #   detector:   ObjectDetection(), must be loaded once
        #               and passed to avoid memory issues
        #   box:        [x1,y1,x2,y2] of box
        #-------------------------------------------

        #restrict our objects to only cell phones
        custom_objects = detector.CustomObjects(cell_phone=True)
        detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(self.folder, self.image),output_image_path=os.path.join(os.getcwd(),"image_detections" , self.image[0:len(self.image)-4]+"_det.jpg"), minimum_percentage_probability=min_prob)
    
        #Note: if no object found, set_error() will still catch error
        if len(detections)!=0:
            #print("-found cell phone")
            box = detections[0]["box_points"]#index=0 is most probable
            self.set_prediction(box)

#=========================================================================

def load_detector():
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    fine_tuned_model_path = os.path.join(os.getcwd(),"logs","000","trained_weights_final.h5")
    detector.setModelPath(fine_tuned_model_path)
    detector.loadModel()#"normal"(default), "fast", "faster" , "fastest" and "flash"
    return detector

class Phones:
    
    def __init__(self):
        self.list = []#list of class instance Phone
        self.error = float()
        self.error_radius = float()
        self.min_prob = float()
    
    def set_model(self, error_radius, min_prob):
        self.error_radius = error_radius
        self.min_prob = min_prob

    def add_phone(self,phone):
        self.list.append(phone)
    
    def initialize_from_txt(self,txt,folder):
        #-------------------------------------------
        #   Objective:  init class by pulling data
        #               from provided txt file
        #-------------------------------------------
        for line in txt:
            self.add_phone(Phone(str(line[0]),float(line[1]),float(line[2]),folder))

    def get_predictions(self):
        #-------------------------------------------
        #   Objective:  get midpoint predictions for
        #               all phones in list
        #-------------------------------------------
        detector = load_detector()#must happen outside loop to avoid memory issues.
        
        for phone in self.list:
            #print("getting predictions for ",phone.image)
            phone.get_prediction(detector,self.min_prob)

    def set_error(self):
        #-------------------------------------------
        #   Objective:  determine if loc_pred is
        #               within the allowed radius
        #-------------------------------------------
        cnt = 0
        for phone in self.list:
            #project to positive quadrant via abs()
            if phone.xmid<0. and phone.ymid<0.:
                phone.error = True
                cnt+=1
            elif self.error_radius < math.sqrt((phone.xmid_pred-phone.xmid)**2+(phone.ymid_pred-phone.ymid)**2):
                phone.error = True
                cnt+=1
            else:
                phone.error = False
        self.error = float(cnt)/float(len(self.list))

#=========================================================================

def read_actual_loc(folder):
    #-------------------------------------------
    #   Objective:  Convert txt file into table
    #-------------------------------------------
    text_file = open(os.path.join(folder, "labels.txt"), "r")
    lines = text_file.read().split("\n")
    table = [line.split(" ") for line in lines[0:len(lines)-2]]
    text_file.close()
    return table

def write_train(txt,folder,side,set_size):
    #-------------------------------------------
    #   Objective:  Write train.txt file used for
    #               fine tuning yolov3.h5. This
    #               serves as a rough automatic
    #               annotation
    #   ----------------------------------------
    #   txt:        [line]
    #   line:       [image, xmid, ymid]
    #   side:       num pixels in box window
    #   set_size:   % of total available training set
    #-------------------------------------------
    def bounds(xmid,ymid,side,path):
        im = Image.open(path)
        xmax,ymax=im.size[0],im.size[1]
        im.close()
        
        xlow =  float(xmid)*xmax-side/2
        xhigh = float(xmid)*xmax+side/2
        ylow =  float(ymid)*ymax-side/2
        yhigh = float(ymid)*ymax+side/2
        
        if xlow<0:xlow=0
        if xhigh>xmax:xhigh=xmax
        if ylow<0:ylow=0
        if yhigh>ymax:yhigh=ymax

        return int(math.floor(xlow)),int(math.floor(xhigh)),int(math.floor(ylow)),int(math.floor(yhigh))

    txt=txt[0:round(set_size*len(txt))]#create subset if desired
    filepath = os.path.join(os.getcwd(),"model_data","train.txt")
    with open(filepath, 'w') as the_file:
        for i,line in enumerate(txt):
            path=os.path.join(folder,line[0])
            xlow,xhigh,ylow,yhigh=bounds(line[1],line[2],side,path)
            #we know from keras-yolo3 that cell phone=67
            if i==len(txt)-1:
                the_file.write(str(path)+' '+str(xlow)+','+str(ylow)+','+str(xhigh)+','+str(yhigh)+','+'67')
            else:
                the_file.write(str(path)+' '+str(xlow)+','+str(ylow)+','+str(xhigh)+','+str(yhigh)+','+'67\n')
        the_file.close()

#=========================================================================

def save_model_parameters(dict):
    filename = 'mp'
    outfile = open(filename,'wb')
    pickle.dump(dict,outfile)
    outfile.close()

def load_model_parameters():
    filename = 'mp'
    infile = open(filename,'rb')
    dict = pickle.load(infile)
    infile.close()
    return dict

class Range:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def rand_val(self):
        return randint(self.min,self.max)

def optimize_object_detection(phones,batch_range,ep_range,mp_range,error_radius,it,th):
    #--------------------------------------------------------------------------------
    #   Objective:      Find an object detection model with error below threshold
    #
    #   Description:    Use ImageAI for object detection, a COCO-pretrained yolo.h5
    #                   model, and keras-yolo3 to fine-tune the pretrained model.
    #                   We optimize through a random parameter space search, and
    #                   stop when we we've reached desired accuracy on training set
    #   -----------------------------------------------------------------------------
    #   phones:         [Phone], holds all data needed for object detection
    #   batch_range:    Range, used for fine tuning in keras-yolo3
    #   ep_range:       Range, (epochs) used for fine tuning in keras-yolo3
    #   mp_range:       Range, min probability needed to detect cell phone
    #   error_radius:   error radius allowed for prediction of center of cell phones
    #   th:             error threshold. For task 30% is most allowed
    #--------------------------------------------------------------------------------
    
    for i in range(1,it):
        
        print("--- optimize_object_detection: it = ",i)
        
        #---Model parameters
        batch_size=batch_range.rand_val()
        epochs=ep_range.rand_val()
        min_prob=mp_range.rand_val()
        
        model_parameters = {"bs":batch_size,"e":epochs,"mp":min_prob}

        #---Fine tune using keras-yolo3
        print("-Fine-tunning yolo3")
        train.main(batch_size,epochs)

        #---Test object detection on training set
        print("-Object detection via ImageAI")
        phones.set_model(error_radius,min_prob)
        phones.get_predictions()
        phones.set_error()

        model_parameters["error"]=phones.error
        print("-Model_parameters: ",model_parameters)

        if phones.error < th:
            save_model_parameters(model_parameters)
            break
    if phones.error < th:
        #if error is below threshold break loop, return error. Note, fine-tuned model is saved, we must save the correct mp value to be used in the find_phone.py
        print("-Found satisfactory model")
    else:
        print("No models were found beneath error threshold. Try changing parameter ranges, error_threshold, or running more iterations.")

#=========================================================================

if __name__ == "__main__":
    
    print("\n---Calling: train_phone_finder.py---\n")

    #---Take in arguments at command line
    find_phone_path = sys.argv[1]
    
    #---Convert given data to form used for training
    txt = read_actual_loc(find_phone_path)
    write_train(txt,find_phone_path,40,1.0)
    
    #---Create classes
    phones = Phones()
    phones.initialize_from_txt(txt,find_phone_path)
    
    #---Train and optimize detection model
    batch_range=Range(9,10)
    ep_range=Range(2,4)
    mp_range=Range(10,15)
    error_radius=0.05
    it=10
    th=0.1
    
    optimize_object_detection(phones,batch_range,ep_range,mp_range,error_radius,it,th)
    
    print("\n---End: train_phone_finder.py---\n")



