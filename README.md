# JetsonNanoProjectDoane
Pneumonia Detection Using Chest X-Ray Images on NVIDIA Jetson Orin Nano 

This project detects pneumonia from chest X-ray images using a retrained ResNet-18 learning model. The goal of this detection system is to assist healthcare professionals by making pneumonia diagnosis more efficient, reducing the time required to analyze X-ray images, and works as an additional tool to support clinical decisions.
Pneumonia is a serious lung infection that affects millions of people each year. An AI detection system like this has many possible applications in hospitals, clinics, and other healthcare settings. While this model is intended as a decision-support tool it demonstrates the potential of AI running on the NVIDIA Jetson Orin Nano for real-time medical image analysis.
<img width="1627" height="681" alt="Screenshot 2026-07-02 135556" src="https://github.com/user-attachments/assets/fa4731f2-e0ab-413f-a1a1-613707c833e1" />
(Real photo of Pneumonia detection)


# The Algorithm
The process of detecting pneumonia in lung scans started by being trained through a series of images that are healthy and that have pneumonia. The retrained ResNet-18 tool allows AI to learn what a lung scan with pneumonia looks like versus one without. After the model is trained it is then tested to see if it successfully can detect different lung scans based on the knowledge it learned. The only classification options for the images are “normal” and “pneumonia”. 


# Running this project
1. Make sure that both the Jetson Inference library and Python3 are installed on your Jetson Nano.


2. Download the data set, Link to dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia


3. To unzip the file type sudo apt install unzip - Then, type unzip


4. Go to the Docker container by going to the jetson-inference folder. Type cd jetson-inference - Then, type ./docker/run.sh


5. Inside the Docker container, change directories. Type cd jetson-inference/python/training/classification, then train the AI with the dataset by doing python3 train.py --model-dir=models/ data/


6. Generate resnet18.onnx to process the images for testing. Type: python3 onnx_export.py --model-dir=models/


7. Set the NET and DATASET variables. Type NET=models/ - Then, type DATASET=data/ - 


8. Make a place for the output files. Type $DATASET/test_output_<Name of Category 1> $DATASET/test_output_<Name of Category 2>


9. Test 1 image. Copy the name of an image in the Category 1 folder of your test folder. Type imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/<Name of Category 1>/.jpg .jpg


10. Type imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/<Category 1 Name> $DATASET/test_output_<Category 1 Name> - 
 To test all category images type imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/<Category 2 Name> $DATASET/test_output_<Category 2 Name> -


11. Go to jetson-inference --> python --> training --> classification --> data - You should see 2 output folders. After clicking the images there should be a text box showing the exact accuracy of the AI.

(Video of how it works)
https://youtu.be/uYo9yej3qNU




