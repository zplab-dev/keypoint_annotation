import os, random, time, copy

import numpy as np
import os.path as path
import scipy.io as sio

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

def train_reg(model, dataloaders, dataset_sizes, loss_1_to_2, optimizer, scheduler, 
                start_epo = 0,num_epochs=25, work_dir='./', device='cpu'):
    
    log_filename = os.path.join(work_dir,'train.log')    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(start_epo ,num_epochs):        
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        fn = open(log_filename,'a')
        fn.write('\nEpoch {}/{}\n'.format(epoch+1, num_epochs))
        fn.write('--'*5+'\n')
        fn.close()

        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print(phase)
            fn = open(log_filename,'a')        
            fn.write(phase+'\n')
            fn.close()
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else: model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_acc = 0

            # Iterate over data.
            img, classes, out = None, None, None
            iterCount,sampleCount = 0, 0
            for sample in dataloaders[phase]:
                img, labels = sample  
                img = img.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train 
                acc = 0
                preds = None          
                with torch.set_grad_enabled(phase=='train'):
                    if phase=='train':  # backward + optimize only if in training phase
                        model.train()                        
                        output = model(img)
                        _, preds = torch.max(output, 1)                         
                        loss = loss_1_to_2(output, labels.data)
                        loss.backward()
                        #print("labels: ", labels.data)
                        #print("preds: ", preds)
                        acc = torch.sum(preds == labels.data)
                        optimizer.step()
                        
                        
                    else: 
                        model.eval()                        
                        output = model(img)    
                        _, preds = torch.max(output, 1)                      
                        loss = loss_1_to_2(output, labels)
                        acc = torch.sum(preds == labels)


                # statistics  
                iterCount += 1
                sampleCount += img.size(0)                                
                running_loss += loss.item() * img.size(0) 
                running_acc += acc
                accprint2screen_avgLoss = running_acc.double()/sampleCount
                print2screen_avgLoss = running_loss/sampleCount
                                               
                if iterCount%50==0:
                    print('\t{}/{} loss: {:.4f} \t acc: {:.4f}'.format(iterCount, len(dataloaders[phase]), print2screen_avgLoss, accprint2screen_avgLoss))
                    fn = open(log_filename,'a')        
                    fn.write('\t{}/{} loss: {:.4f} \t acc: {:.4f}\n'.format(iterCount, len(dataloaders[phase]), print2screen_avgLoss, accprint2screen_avgLoss))
                    fn.close()
  
            epoch_loss = running_loss / dataset_sizes[phase]
            accepoch_loss = running_acc.double() / dataset_sizes[phase]
                                
            print('\tloss: {:.6f}\t acc: {:.6f}'.format(epoch_loss, accepoch_loss))
            fn = open(log_filename,'a')
            fn.write('\tloss: {:.6f}\t acc: {:.6f}\n'.format(epoch_loss, accepoch_loss))
            fn.close()
            
            plot_output(img, labels, preds, epoch, phase, work_dir)

            # deep copy the model
            cur_model_wts = copy.deepcopy(model.state_dict())
            path_to_save_paramOnly = os.path.join(work_dir, 'epoch-{}.paramOnly'.format(epoch+1))
            torch.save(cur_model_wts, path_to_save_paramOnly)
            
            if phase=='val' and epoch_loss<best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                np.save('best', best_loss)
                
                path_to_save_paramOnly = os.path.join(work_dir, 'bestValModel.paramOnly')
                torch.save(best_model_wts, path_to_save_paramOnly)
                #path_to_save_wholeModel = os.path.join(work_dir, 'bestValModel.wholeModel')
                #torch.save(model, path_to_save_wholeModel)
                
                file_to_note_bestModel = os.path.join(work_dir,'note_bestModel.log')
                fn = open(file_to_note_bestModel,'a')
                fn.write('The best model is achieved at epoch-{}: loss{:.6f}.\n'.format(epoch+1,best_loss))
                fn.close()
                
        scheduler.step()       
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    fn = open(log_filename,'a')
    fn.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    fn.close()
   
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def plot_output(imgList, labels, out, epoch, phase, work_dir='./'):
    figWinNumHeight, figWinNumWidth, subwinCount = 4, 4, 1
    plt.figure(figsize=(22,20), dpi=88, facecolor='w', edgecolor='k') # figsize -- inch-by-inch
    plt.clf()

    for sampleIndex in range(min(4, len(imgList))):
        # visualize image
        plt.subplot(figWinNumHeight,figWinNumWidth,subwinCount)
        subwinCount += 1
        image = imgList[sampleIndex].cpu().numpy()#.squeeze().transpose((1,2,0))      
        plt.imshow(image[0], cmap='gray')    
        plt.axis('off')
        plt.title('True: {}  Predicted: {}'.format(labels[sampleIndex], out[sampleIndex]))

    save_path = os.path.join(work_dir, ('epoch '+str(epoch)+' output '+phase+'.png'))
    plt.savefig(save_path)
    plt.close()

def init_model(pretrained=True):
    model_ft = models.resnet34(pretrained=pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    return model_ft

