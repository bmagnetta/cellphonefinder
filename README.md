=============================================================
	README
=============================================================

Author: Brad Magnetta

Description: The task is to detect
cell-phones in provided images by any means necessary. Because the provided dataset is very small
we used a pre-trained model as a foundation, and fine-tuned it using the provided data. This is also needed because our small dataset contains cellphones that look different from the cellphones in the pre-trained model.
Our approach has proven to be very accurate, easily fined tuned for near perfect prediction of
the center positions of each phone in the training images, for the provided center tolerance. It
should be noted that because a visual inspection of the training data revealed very little other objects
in the images, we have set the minimum probability for object detection to be fairly low ~(20%-30%).

------------------------------
	References
------------------------------

- keras-yolo3:
	- fine-tune a .h5 yolov3 model
	- https://github.com/qqwweee/keras-yolo3
- imageai:
	- Object detection using trained model in .h5 format
	- https://github.com/OlafenwaMoses/ImageAI

------------------------------
	Comments on Files
------------------------------

 - Google Drive shared folder:
        - Contains image data, pre-trained models, virtual environment needed to run this code
        - Move virtual environment to the folder containing the below files or create your own.


- manage.py

    - After downloading the shared folder in the link above, change the base_path parameter to the location of your download.

- train_phone_finder.py

	- Test for dependencies using module check from check_dep.py.

	- Take in a command line argument; a folder path containing images and known 
	cellphone center positions for model training.
	
		- python3 train_phone_finder.py path/to/training_data

	- Populate given data in classes Phone and Phones. These classes contain
	modules for object detection.

	- Organize txt file containing annotation information for training data
	(/model_data/train.txt)

	- Take a COCO pre-trained YOLO model (/model_data/yolo_weights_coco.h5), and 
	fine-tune using keras-yolo3 (train.py & /yolo3) and the given data 
	(/model_data/train.txt). 2 sweeps, the first unfreezes last 3 layers and
	retrains keras model, the second unfreezes all layers and continues training.

	- take fine-tuned model (/logs/000/trained_weights_final.h5) and use
	in imageai, for object detection. Probe model parameter space via random
	configurations and a minimum allowable performance on training dataset. Record
	error on detected phone centers of satisfactory model configuration, save model 
	parameters. Store detections in new images and save (/image_detections).


- find_phone.py

	- Test for dependencies using module check from check_dep.py.

	- Suppress harmless warnings from printing in command line

	- Take in a command line argument; an image path.
	
		- python3 find_phone.py path/to/image

	- Use modules from train_phone_finder.py: Populate given data in class Phone.
	Predict the center of detected phone in image and print center in command line.

------------------------------
	Future Work and Desired Collaboration
------------------------------

- Generalize this code to detect other images using fine tuned models.

- Generalize this code to other pre-trained models.

- Apply this work to detect cell-phones in video.

