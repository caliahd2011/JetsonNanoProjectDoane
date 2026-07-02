# JetsonNanoProjectDoane
Pneumonia Detection Using Chest X-Ray Images on NVIDIA Jetson Orin Nano 

This project detects pneumonia from chest X-ray images using a retrained ResNet-18 learning model. The goal of this detection system is to assist healthcare professionals by making pneumonia diagnosis more efficient, reducing the time required to analyze X-ray images, and works as an additional tool to support clinical decisions.
Pneumonia is a serious lung infection that affects millions of people each year. An AI detection system like this has many possible applications in hospitals, clinics, and other healthcare settings. While this model is intended as a decision-support tool it demonstrates the potential of AI running on the NVIDIA Jetson Orin Nano for real-time medical image analysis.

(Link of project working)

# The Algorithm
The process of detecting pneumonia in lung scans started by being trained through a series of images that are healthy and that have pneumonia. The retrained ResNet-18 tool allows AI to learn what a lung scan with pneumonia looks like versus one without. After the model is trained it is then tested to see if it successfully can detect different lung scans based on the knowledge it learned. The only classification options for the images are “normal” and “pneumonia”. 


# Running this project
Make sure that resnet and googlenet have been installed. This can be done by doing this command: sudo apt-get install libpython3-dev python3-numpy, changing to the docker container (from the jetson-inference folder) by doing ./docker/run.sh, changing directories to jetson-inference/tools, and doing then doing this command: ./download-models.sh You should see a menu pop up. Then, make sure that there is a star next to resnet and googlenet. This signifies that resnet and googlenet are downloaded. Also, make sure that PyTorch does not have a star next to it, as it does not need to be downloaded.
Create your dataset by collecting photos. You can use a website like "Kaggle" to find some collections of photos. You will need to have 2 categories of photos. When trained, the AI will sort images into the 2 categories. I downloaded photos of skin that corresponded to people that were healthy or unhealthy. So, my two categories were healthy skin and unhealthy skin. Once you have your photos, create a folder for your dataset. I titled mine "newproject." In this folder, create 3 more folders with titles "test", "train" and "val"(validation). Use lowercase letters. In each of these folders, create 1 folder with the name of one of your categories and 1 folder with the name of the other category. Then, cut and paste your images into the 2 folders of the test, train, and val folders. Make sure that you paste the images of category 1 to test, train, and val's category 1 folders and do the same for category 2. For example, my category 1 was "healthy." So, I pasted my "healthy" photos to the test, train, and val "healthy" folders. Note that about 5% of your photos should be pasted into val, about 10% or so in test, and the remaining 85% in train. Also, create a labels.txt file in your dataset folder (which was the one I titled "newproject"). Type the name of 1 category, press enter, then type the name of the other category. Use lowercase letters for the labels.txt file. Turn the dataset folder into a zip file and then upload the file to MediaFire. Click on the folder in MediaFire, and then copy the link address of the file (AKA copy the link of the BLUE download button).
Go into your Linux Terminal/PuTTY. Download the dataset by typing wget
Unzip the file. This can be done by typing sudo apt install unzip - Then, type unzip
Go to the Docker container by first going to the jetson-inference folder. This can be done by doing cd jetson-inference - Then, type ./docker/run.sh
Inside the Docker container, change directories. Type cd jetson-inference/python/training/classification - Now, we are in the classification folder. Train the AI with the dataset by doing python3 train.py --model-dir=models/ data/
Not changing directories, generate resnet18.onnx to process the images we want to test. This can be done by typing: python3 onnx_export.py --model-dir=models/
Still not changing directories, set the NET and DATASET variables. Type NET=models/ - Then, type DATASET=data/ - These variables will be used in the command to process our images.
Make a place for our output files to go. Type $DATASET/test_output_<Name of Category 1> $DATASET/test_output_<Name of Category 2>
Test 1 image. Copy the name of an image in the Category 1 folder of your test folder. Type imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/<Name of Category 1>/.jpg .jpg
Test the rest! Type imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/<Category 1 Name> $DATASET/test_output_<Category 1 Name> - This will test all Category 1 images.
Then, type imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/<Category 2 Name> $DATASET/test_output_<Category 2 Name> - This will test all Category 2 images.
Look at the results!! One way to see your results is to go to your JetsonNano file explorer. Go to jetson-inference --> python --> training --> classification --> data - You should then see your 2 output folders! Click on each to see how accurate your AI was! On each image, there will be a text box indicating how close the AI believes your image is related to one of your 2 categories.
# BONUS TIP To move your entire project, in you Jetson Nano file explorer, create a folder. In that folder, copy and paste your data and models folders of your dataset folder to the new folder.
Here is a video detailing how my project works!




