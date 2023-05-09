
#%% load the background
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

#%% load and prepare the dataset
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
dataset2use = "normalized_to_onlyH"

if dataset2use == "original":
    data_dir = '/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/patchCamelyon/patches/original'
elif dataset2use == "normalized_to_HE":
    data_dir = '/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/patchCamelyon/patches/normalized_to_HE'
elif dataset2use == "normalized_to_tumorLymphnode_165":
    data_dir = "/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/patchCamelyon/patches/normalized_to_tumorLymphnode_165"
elif dataset2use == "normalized_to_onlyH":
    data_dir = "/home/cw9/sds_hd/sd18a006/marlen/datasets/stainNormalization/patchCamelyon/patches/normalized_to_onlyH"

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# checks if GPU is available, and then decide accordingly...
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print('GPU-mode is set')
else:
    print('CPU-mode is set')

#%% show the image data
# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
from classification_test.visFunctions import imshow

imshow(out, title=[class_names[x] for x in classes])

#%% define the training function
# why can't that be done within another file -> not understandable
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#%% define the model loading function
def load_model(imodel):
    #Load a pretrained model and reset final fully connected layer.
    if imodel == 'ResNet18':
        model2train = models.resnet18(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'ResNet34':
        model2train = models.resnet34(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'ResNet50':
        model2train = models.resnet50(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'ResNet101':
        model2train = models.resnet101(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'ResNet152':
        model2train = models.resnet152(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'vgg16':
        model2train = models.vgg16(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'vgg19':
        model2train = models.vgg19(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'vgg11':
        model2train = models.vgg11(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'alex':
        model2train = models.alexnet(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'inception':
        model2train = models.inception_v3(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'densenet121':
        model2train = models.densenet121(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'densenet161':
        model2train = models.densenet161(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'densenet169':
        model2train = models.densenet169(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'densenet201':
        model2train = models.densenet201(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'squeeznet':
        model2train = models.squeezenet1_0(pretrained=True)
        model2train.modName = imodel + '_loaded '

    return (model2train)

#%% function for preparing the database
def prep_database(inputSize):

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(inputSize, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.RandomResizedCrop(inputSize, scale=(0.7, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return(dataloaders, dataset_sizes, class_names)

#%% define function for adapting a pretrained model
def adapt_model(model_ft, imodel):
    n_classes = 12

    if imodel.find('vgg') == 0 or imodel.find('alex') == 0:

        n_inputs = model_ft.classifier[6].in_features

        # Add on classifier
        model_ft.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))

        total_params = sum(p.numel() for p in model_ft.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in model_ft.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

    elif imodel.find('densenet') == 0:
        model_ft.classifier = nn.Linear(1024, n_classes)

    elif imodel.find('squeeznet') == 0:
        model_ft.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = n_classes

    elif imodel == 'inception':

        model_ft.aux_logits = False
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_classes)

    else:
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    return(model_ft)

#%% define a list of models
model_list = ['alex',
              'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
              'vgg11', 'vgg16', 'vgg19',
              'inception',
              'densenet121']
model_list = ['ResNet152'] # scheinbar ein Problem mit den densenets ab 161?

#save_path = data_dir + '/trainedModels'

#%% iterate over the model list
for imodel in model_list:

    #%% # Data augmentation and normalization for training
    # Just normalization for validation
    if imodel == 'inception':
        inputSize = 299
    else:
        inputSize = 224

    (dataloaders, dataset_sizes, class_names) = prep_database(inputSize)

    #%% load the model
    model2train = load_model(imodel)
    print(model2train.modName)

    #%% adapt the model
    model_ft = model2train  # for debugging re-loading can be avoided
    model_ft = adapt_model(model_ft, imodel)
    model_ft = model_ft.to('cuda')

    #%% set the training parameter
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    #%% train the model
    model_ft = model_ft.to(device)
    model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
                            num_epochs=50)

    #% save the model
    torch.save(model_ft, data_dir+ '/model_' + imodel + '.pt')


