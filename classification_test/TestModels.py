
#%% load the background
from __future__ import print_function, division
import torch
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
import torch.nn as nn

#%% define the datasets
list_datasets = ['/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/patchCamelyon/patches/original',
                '/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/patchCamelyon/patches/normalized_to_HE',
                '/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/patchCamelyon/patches/normalized_to_tumorLymphnode_165',
                '/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/patchCamelyon/patches/normalized_to_onlyH',
                '/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/tumorLymphnode/patches/size_165/original',
                 '/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/tumorLymphnode/patches/size_165/normalized_to_HE_165',
                 '/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/tumorLymphnode/patches/size_165/normalized_to_camelyon_165',
                '/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/tumorLymphnode/patches/size_165/normalized_to_onlyH_165'
                 ]
list_dataset_names = ['camelyon_ori', 'camelyon_to_HE', 'camelyon_to_tL', 'camelyon_to_H',
                      'tumorLymphnode_ori', 'tumorLymphnode_to_HE', 'tumorLymphnode_to_ca', 'tumorLymphnode_to_H']

list_models = ['/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/patchCamelyon/patches/original/model_ResNet152.pt',
            '/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/patchCamelyon/patches/normalized_to_HE/model_ResNet152.pt',
            '/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/patchCamelyon/patches/normalized_to_tumorLymphnode_165/model_ResNet152.pt',
            '/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/patchCamelyon/patches/normalized_to_onlyH/model_ResNet152.pt' ]

list_model_names = ['ResNet_original', "ResNet_normalized_to_HE", "ResNet_normalized_to_tumorLymphnode", "ResNet_normalized_to_H"]

#%% iterate over all datasets (and later over all models)
list_model = []
list_dataset = []
list_kappa = []
list_accuracy = []
list_loss = []

for idataset, tdataset in enumerate(list_datasets):
    #print(idataset)

    #%% define the folder
    if tdataset.find("patches") > 0:
        dataset2use = "val"
    else:
        dataset2use = 'test'

    # %%define the function to get the data
    def get_datatransform(inputSize, data_dir):

        data_transforms = {
            dataset2use: transforms.Compose([
                transforms.Resize([inputSize, inputSize]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in [dataset2use]}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                      shuffle=False, num_workers=4)
                       for x in [dataset2use]}

        return(data_transforms, image_datasets, dataloaders)

    #%% prepare the transformations and the dataset
    data_transforms , image_datasets, dataloaders= get_datatransform(259, tdataset)

    class_names = dataloaders[dataset2use].dataset.classes
    nb_classes = len(class_names)
    confusion_matrix = torch.zeros(nb_classes, nb_classes)

    #%% visualize the input data (to look if evey class is evenly)
    class_names =  ['normal', 'tumor']

    df = pd.DataFrame(dataloaders[dataset2use].dataset.samples)
    df.columns = ['file', 'class_nr']

    df.class_nr = np.array(df.class_nr)

    class_labels = ['NaN' for x in range(df.shape[0])]
    for i in range(0,df.shape[0]):
        class_labels[i] = class_names[df.class_nr[int(i)]]
    df = df.assign(class_labels = class_labels)
    sns.set_palette("Set1", n_colors = 12)
    sns.countplot(df.class_labels)
    plt.xlabel('Pattern')
    plt.ylabel('Count [n]')
    plt.savefig('DataBase_' + dataset2use + '.jpg')
    plt.show()
    plt.close()

    n_normal = sum(map(lambda x : x == "normal", class_labels))
    n_tumor = sum(map(lambda x: x == "tumor", class_labels))
    print("n = " + str(n_normal) + " tiles without and n = " + str(n_tumor) + " tiles with tumor.")

    #%% iterate over the models
    from sklearn.metrics import cohen_kappa_score
    from sklearn.metrics import accuracy_score
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n = 0
    df_values = pd.DataFrame(list(range(0,len(dataloaders[dataset2use].sampler.data_source.imgs))))

    for imodel, tmodel in enumerate(list_models):
        print(imodel)

        #%% prepare the dataset
        inputSize = 224
        data_transforms, image_datasets, dataloaders = get_datatransform(inputSize, tdataset)

        #%% apply model on test data set (and get a confusion matrix)
        model_ft = torch.load(tmodel)
        model_ft.eval()
        vector_prd = []
        vector_exp = []

        with torch.no_grad():
            for i, (inputs, classes) in enumerate(dataloaders[dataset2use]):
                inputs = inputs.to(device)
                classes = classes.to(device)
                outputs = model_ft(inputs)
                _, preds = torch.max(outputs, 1)

                if i == 0:
                    outputs_matrix = outputs
                else:
                    outputs_matrix = torch.cat((outputs_matrix, outputs), 0)

                vector_prd = vector_prd + preds.view(-1).cpu().tolist()
                vector_exp = vector_exp + classes.view(-1).cpu().tolist()

        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        for x, y in zip(vector_exp, vector_prd):
            confusion_matrix[y, x] += 1

        loss_function = nn.CrossEntropyLoss()
        loss_value = loss_function(outputs_matrix.to('cpu'), torch.tensor(vector_exp))
        print(confusion_matrix)

        #%% calcualte the comparison values
        list_model.append(list_model_names[imodel])
        list_dataset.append(list_dataset_names[idataset])
        list_kappa.append(cohen_kappa_score(vector_prd, vector_exp))
        list_accuracy.append(accuracy_score(vector_prd, vector_exp))
        list_loss.append(loss_value.tolist())
        print('Kappa-value: ' + str(list_kappa[-1]))
        print('Accurary-value: ' + str(list_accuracy[-1]))

        #%% plot a confusion matrix
        matrix2plot = confusion_matrix.numpy()
        matrix2plot = matrix2plot.astype(int)

        ax = sns.heatmap(matrix2plot,
                         annot = True, linewidths=5, annot_kws={"size": 10},
                         xticklabels=class_names, yticklabels=class_names,
                         cmap = "Blues")
        plt.xlabel('Ground Truth')
        plt.ylabel('Model ' + list_model[-1] + " on " + list_dataset[-1])
        plt.savefig('ConfMat_' +'Model ' + list_model[-1] + " on " + list_dataset[-1] + '.jpg')
        plt.show()
        plt.close()

#%% make a dataframe
df = pd.DataFrame(list(zip(list_model, list_dataset, list_kappa)), columns=['model', 'data', 'kappa'])
df = df.pivot_table(index = ["model"], columns = ["data"], values = "kappa")
df.to_csv('/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/table.csv')
df.to_excel('/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/table.xlsx')
with open('/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/table.tex', 'w') as tf:
    tf.write(df.to_latex())