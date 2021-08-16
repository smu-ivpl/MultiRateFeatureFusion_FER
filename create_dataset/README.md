데이터셋 만드는 방법

1. preprocess_image.py
전체 이미지 시퀀스에 전처리를 한다 

2. choose_frame_minimum_ckp_14.py
(in CKP 데이터베이스)
전처리된 이미지 시퀀스에서 minimum overlapped frame structure를 사용하여 3개, 5개, 7개의 이미지 프레임을 뽑고, 
이를 14배 augmentation 시킨 후, .npy 파일로 저장한다.

3. move_neutral_smf_tot.py
먼저 neutral 표정의 데이터셋을 neutral 표정의 폴더(0)으로 분류하고,
3개, 5개, 7개 프레임수로 이루어진 데이터셋을 분류한다.

4. npy_merge.py
저장된 .npy 파일들을 merge 시켜 최종적으로 실험에 쓰이는 데이터셋을 만든다.
