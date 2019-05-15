#=========================================================================
#   Brain Corp. Code Sample: 2019 Summer Intern - R&D Machine Learning
#   Author: Brad Magnetta
#-------------------------------------------------------------------------
#   Goals:  1) predict the center of a cell phone in a provided image
#
#=========================================================================
import sys
import os

#The following is needed to avoid keras command print "using tensorflow backend"
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import check_dep
check_dep.check()

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#ignores AVX2 message
tf.logging.set_verbosity(tf.logging.FATAL)#ignores dep. message

import train_phone_finder
from imageai.Detection import ObjectDetection
#=========================================================================

if __name__ == "__main__":
    #---Take in image path
    image_path = sys.argv[1]
    
    #---Initialize phone class. Has internal functions needed for object detection
    phone = train_phone_finder.Phone(image_path,-1.,-1.,"")
    model_parameters = train_phone_finder.load_model_parameters()

    #---Load detector model
    detector = train_phone_finder.load_detector()

    #---Detect cell phone in image, print predicted center in float(),space,float() format
    phone.get_prediction(detector,model_parameters["mp"])
    print(float(phone.xmid_pred)," ",float(phone.ymid_pred))


