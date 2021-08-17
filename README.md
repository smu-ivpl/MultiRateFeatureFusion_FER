# Paper
"A Robust Facial Expression Recognition Algorithm Based on Multi-Rate Feature Fusion Scheme"



# Model Stucture
![figure1_3](https://user-images.githubusercontent.com/47851661/129718320-2da7dd87-f1ab-4b85-801a-d49272aaedc9.png)



# Installation (Environment)
- python 3.7
- pip install tensorflow-gpu==2.1.0
- pip install keras==2.2.4
- pip install theano
- pip install opencv-python
- pip install matplotlib
- pip install numpy
- pip install keras-self-attention
- Install ‘FaceBoxes’ module’s environment for face detection
   ( https://github.com/sfzhang15/FaceBoxes )
- Install ’SAN’ module’s environment for landmark detection
   ( https://github.com/D-X-Y/landmark-detection )



# Train and Test
concat_train_test.py
- first input = selected dataset you want to train and test from the above list
- second input = iteration of training & testing preprocessing list (pre = ['pre', 'lbp', 'norm', 'normlbp'])

* example

first input = 1 -> select 'MMI_minimum dataset'

second input = 3 -> train & test selected database of preprocessed, lbp, normalized dataset
