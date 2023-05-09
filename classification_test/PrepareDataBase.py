
#%% prepare the background
import os
import numpy as np
import shutil
import glob

#%% Creating Train / Val / Test folders (One time use)
dataset = "camelyon"
if dataset == "camelyon":
    root_dir = '/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/patchCamelyon/patches'
    folders = ['original',
               'normalized_to_HE',
               'normalized_to_tumorLymphnode_165']
    folders = ['normalized_to_onlyH']
elif dataset == "tumorLymphnode":
    root_dir = '/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/tumorLymphnode/patches/size_165'
    folders = ['original',
               'normalized_to_HE_165',
               'normalized_to_camelyon_165']
    folders = ['normalized_to_onlyH_165']

classes_dir = ['/tumor',
               '/normal']

class_names =  ['/tumor',
                '/normal']

val_ratio = 0.25
test_ratio = 0

#%% iterate over it
for ifolder in folders:

    for i in range(0, len(classes_dir)):

        #% counter section
        print('folder ' + classes_dir[i] + ' started')

        #% prepare the directories
        trainFolder = root_dir + "/" + ifolder + '/train' + class_names[i]
        if os.path.exists(trainFolder):
            shutil.rmtree(trainFolder)
        os.makedirs(trainFolder)
        # validation folder
        valFolder = root_dir + "/" + ifolder +'/val' + class_names[i]
        if os.path.exists(valFolder):
            shutil.rmtree(valFolder)
        os.makedirs(root_dir +"/" + ifolder + '/val' + class_names[i])
        # test folder
        testFolder = root_dir + "/" + ifolder +'/test' + class_names[i]
        if os.path.exists(testFolder):
            shutil.rmtree(testFolder)
        os.makedirs(root_dir + "/" + ifolder +'/test' + class_names[i])

        #% prepare the data
        # Creating partitions of the data after shuffeling
        src = root_dir + "/" + ifolder + classes_dir[i]  # Folder to copy images from

        allFileNames = glob.glob(src + '/*.png')
        np.random.shuffle(allFileNames)
        train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                                  [int(len(allFileNames) * (1 - val_ratio + test_ratio)),
                                                                   int(len(allFileNames) * (1 - test_ratio))])

        train_FileNames = [name for name in train_FileNames.tolist()]
        val_FileNames = [name for name in val_FileNames.tolist()]
        test_FileNames = [name for name in test_FileNames.tolist()]

        print('Total images: ', len(allFileNames))
        print('Training: ', len(train_FileNames))
        print('Validation: ', len(val_FileNames))
        print('Testing: ', len(test_FileNames))

        #% Copy-pasting images
        for name in train_FileNames:
            shutil.copy(name, root_dir + "/" + ifolder + '/train' + class_names[i])

        for name in val_FileNames:
            shutil.copy(name, root_dir + "/" + ifolder + '/val' + class_names[i])

        for name in test_FileNames:
            shutil.copy(name, root_dir + "/" + ifolder + '/test' + class_names[i])

        #% counter section
        print('folder ' + classes_dir[i] + ' finished')
        print('folder ' + class_names[i]+ ' finished')


