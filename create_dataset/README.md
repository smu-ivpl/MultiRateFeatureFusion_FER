# Making DATASET

### Order

#### 1. preprocess_image.py
   : preprocess all images in image sequence   
 

```
python preprocess_image.py /home/000/MultiRateFeatureFusion_FER/Database/CKP/emotion_labels /home/000/MultiRateFeatureFusion_FER/Database/CKP/cohn-kanade-images/ /home/000/MultiRateFeatureFusion_FER/Database/CKP/preprocessed_images/ 0 9 128 128]
```
- argv[0] : label_main_dir, directory where the CKP emotion label located [/home/000/MultiRateFeatureFusion_FER/Database/CKP/emotion_labels]
- argv[1] : image_main_dir, directory where the CKP images located [/home/000/MultiRateFeatureFusion_FER/Database/CKP/cohn-kanade-images/]
- argv[2] : emotions_dir, directory where you want to save the preprocessed images [/home/000/MultiRateFeatureFusion_FER/Database/CKP/preprocessed_images/]
- argv[3] : neutral_label, label of neutral expression [0]
- argv[4] : min_seq_len, the minimum number required to create a dataset [9]
- argv[5] : t_height, resized height [128]
- argv[6] : t_width, resized width [128]
- Also, need to change 'model', 'caffe_root', 'model_def', and 'model_weights' in the code
     

#### 2. choose_frame_minimum_ckp_14.py
   : (in CKP database) choose multi-frames through minimum overlapped frame structure and augment 14 times. Then save as .npy file.   

```
python choose_frame_minimum_ckp_14.py /home/000/MultiRateFeatureFusion_FER/Database/CKP/preprocessed_images/ /home/000/MultiRateFeatureFusion_FER/Database/CKP/augmented_images/ 0 128 128]
```
- argv[0] : image_main_dir, directory where the preprocessed CKP image seqences located [/home/000/MultiRateFeatureFusion_FER/Database/CKP/preprocessed_images/]
- argv[1] : emotions_dir, directory where you want to save the multi-frame & augmented images [/home/000/MultiRateFeatureFusion_FER/Database/CKP/augmented_images/]
- argv[2] : neutral_label, label of neutral expression [0]
- argv[3] : t_height, resized height [128]
- argv[4] : t_width, resized width [128]


#### 3. move_neutral_smf_tot.py
   : classify(neutral emotion and multi-frames(3,5,7)) and put then into the right folders   


change the arguments in the code
- d : choose the database from the database list [dataset = ['FERA', 'MMI', 'CKP', 'AFEW']]
- t : choose the dataset which is consists of test, train+validation or test+train+validation dataset[tot = ['Test', 'Train', 'ALL']]
- a : number how many time the dataset is augmented [aug = ['14', '8', '2', '4']]
- m : choose the dataset which is made using minimum overlapped frame structure or not [mo = ['Minimum', 'Overlapped']]
- folder_path : path where you want to classify the files are located


#### 4. npy_merge.py
   : merge .npy files to make datasets which are fed into the network   


change the arguments in the code
```
make_npy(dataset, aug, tot, mo, smf, preprocess)
```
    
- dataset : choose the database from the database list [dataset = ['FERA', 'MMI', 'CKP', 'AFEW']]
- aug : number how many time the dataset is augmented [aug = ['14', '8', '2', '4']]
- tot : choose the dataset which is consists of test, train+validation or test+train+validation dataset [tot = ['Test', 'Train', 'ALL']]
- mo : choose the dataset which is made using minimum overlapped frame structure or not [mo = ['Minimum', 'Overlapped']]
- smf : choose how may frames the dataset consists of [smf = ['s','m','f']
- preprocess : choose the preprocessed method which you want to make a final dataset [preprocess = ['pre','lbp','norm','normlbp']]
- main_path : you need to change the main_path
