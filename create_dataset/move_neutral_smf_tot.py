import os
import shutil

def last_8chars(x):
    """Function that aids at list sorting.
      Args:
        x: String name of files.
      Returns:
        A string of the last 8 characters of x.
    """

    return (x[-8:])

aug = ['14', '8', '2', '4'] # '14' or '8'
tot = ['Test', 'Train', 'ALL'] # 'Train', 'Test' or 'ALL'
mo = ['Minimum', 'Overlapped'] # 'Minimum' or 'Overlapped'
dataset = ['FERA', 'MMI', 'CKP', 'AFEW']  #'FERA' # 'MMI', 'CKP', 'AFEW'

# choose from the lists
d = dataset[3]
t = tot[0]
a = aug[2]
m = mo[0]

folder_path = '/home/sjpark/FER/AFEW/{}_{}_{}/{}/'.format(d, t, a, m)
folder_list = [
            x
            for x in sorted(os.listdir(folder_path), key=last_8chars)
            ]

for index, folder in enumerate(folder_list):

    f_new_path = '{}{}/fast/'.format(folder_path, folder)
    m_new_path = '{}{}/middle/'.format(folder_path, folder)
    s_new_path = '{}{}/slow/'.format(folder_path, folder)
    n_dir_path = '{}{}/0/'.format(folder_path, folder)
    dir_path = '{}{}/'.format(folder_path, folder)

    # if not os.path.exists(f_new_path):
    #     os.makedirs(f_new_path)
    # if not os.path.exists(m_new_path):
    #     os.makedirs(m_new_path)
    # if not os.path.exists(s_new_path):
    #     os.makedirs(s_new_path)
    if not os.path.exists(n_dir_path):
        os.makedirs(n_dir_path)

    if d=='CKP' or d=='MMI' or d=='MMI_All':
        emotions1 = ['1','3','4','5','6','7']
        emotions2 = ['0','1','3','4','5','6','7']
        
        # move neutral expression datasets
        for e in emotions1:
            npy_path = os.path.join(dir_path, e)
            npys = os.listdir(npy_path)
            for npy in npys:
                npy_name = npy.split("-")[0]

                if npy_name == 'n':
                    moving_npy = os.path.join(npy_path, npy)
                    shutil.move(moving_npy, n_dir_path)
                    print(npy)

    if d == 'FERA':
        emotions2 = ['0','1','4','5','6']
    if d == 'AFEW':
        emotions2 = ['0', '1', '3', '4', '5', '6', '7']
    
    # move f,m,s datasets
    for e in emotions2:
        fast_new_path = os.path.join(f_new_path, e)
        middle_new_path = os.path.join(m_new_path, e)
        slow_new_path = os.path.join(s_new_path, e)

        if not os.path.exists(fast_new_path):
            os.makedirs(fast_new_path)
        if not os.path.exists(middle_new_path):
            os.makedirs(middle_new_path)
        if not os.path.exists(slow_new_path):
            os.makedirs(slow_new_path)

        npy_path = os.path.join(dir_path, e)
        npys = os.listdir(npy_path)
        for npy in npys:
            npy_name1 = npy.split("-")[0]
            npy_name2 = npy.split("-")[1]

            if npy_name1 == "f":
                moving_npy = os.path.join(npy_path, npy)
                shutil.move(moving_npy, fast_new_path)
                print(npy)

            if npy_name2 == "f":
                moving_npy = os.path.join(npy_path, npy)
                shutil.move(moving_npy, fast_new_path)
                print(npy)

            if npy_name1 == "m":
                moving_npy = os.path.join(npy_path, npy)
                shutil.move(moving_npy, middle_new_path)
                print(npy)

            if npy_name2 == "m":
                moving_npy = os.path.join(npy_path, npy)
                shutil.move(moving_npy, middle_new_path)
                print(npy)

            if npy_name1 == "s":
                moving_npy = os.path.join(npy_path, npy)
                shutil.move(moving_npy, slow_new_path)
                print(npy)

            if npy_name2 == "s":
                moving_npy = os.path.join(npy_path, npy)
                shutil.move(moving_npy, slow_new_path)
                print(npy)
