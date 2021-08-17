# Making DATASET

Order

1. preprocess_image.py

  : preprocess all images in image sequence
 

2. choose_frame_minimum_ckp_14.py

  : (in CKP database) choose multi-frames through minimum overlapped frame structure and augment 14 times. Then save as .npy file.


3. move_neutral_smf_tot.py

  : classify(neutral emotion and multi-frames(3,5,7)) and put then into the right folders


4. npy_merge.py

  : merge .npy files to make datasets which are fed into the network

