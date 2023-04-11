import os
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.cuda.amp import autocast as autocast
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import warnings
from torchvision import transforms

warnings.filterwarnings("ignore", category=FutureWarning)
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_data(train_dir , val_dir , test_dir):

    classes = os.listdir(train_dir)
    train_imgs = []
    valid_imgs = []
    test_imgs = []

    for _class in classes:
        
        for img in os.listdir(train_dir + _class):
            train_imgs.append(train_dir + _class + "/" + img)
        if val_dir :
            for img in os.listdir(val_dir + _class):
                valid_imgs.append(val_dir + _class + "/" + img)
        if test_dir :
            for img in os.listdir(test_dir + _class):
                test_imgs.append(test_dir + _class + "/" + img)

    ClassInt = {classes[i] : i for i in range(len(classes))}

    return train_imgs , valid_imgs , test_imgs , ClassInt


def img_transform(img):

    tranfromer = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(64, padding=8),
        transforms.ToTensor(),
        transforms.Normalize([0.439, 0.459, 0.406], [0.185, 0.186, 0.229]),
    ])

    return tranfromer(img)

def train( dataloader , model , loss_function , optimizer , lr_scheduler , scaler   ):

    model.train()
    accumulation_step = 2

    correct    = 0
    train_loss = 0
    avg_loss   = 0

    count = tqdm(dataloader)

    for i , (im, label) in enumerate(count):

        im, label = im.to(device), label.to(device)
        shape     = im.shape[0]
        img_size  = im.shape[3]
        
        try:
            im = im.view(shape,-1,img_size,img_size)
        except:
            pass

        with autocast():
            pred = model(im)
            # print(pred.shape)
            # print(label.shape)
            loss = loss_function(pred, label)

        train_loss += loss.item()
        loss /= accumulation_step
        scaler.scale(loss).backward()
        
        p = F.softmax(pred, dim=1)
        correct += (p.argmax(1) == label.argmax(1)).type(torch.float).sum().item()
        
        if((i+1) % accumulation_step == 0 or (i+1 == len(dataloader))):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            avg_loss = float(train_loss / (i+1))
            lr_scheduler.step(loss.item())
            count.set_postfix({'lr':round(lr_scheduler.optimizer.param_groups[0]['lr'],10)})
            count.set_description('loss:{:.5f}'.format(avg_loss))

    size = len(dataloader.dataset)
    print("Training: loss:{:.5f}, Acc:{:.5f}%, lr:{}".format(avg_loss, correct/size*100, lr_scheduler.optimizer.param_groups[0]['lr']))
    return avg_loss , correct/size

def valid(dataloader , model , loss_function):

    size = len(dataloader.dataset)
    test_loss = 0
    correct = 0

    y_true_list = []
    y_pred_list = []

    precision = 0
    recall    = 0
    f1        = 0

    with torch.no_grad():
        for im, label in tqdm(dataloader):

            im, label = im.to(device), label.to(device)
            shape = im.shape[0]
            img_size = im.shape[3]

            try:
                im = im.view(shape,-1,img_size,img_size)
            except:
                continue

            pred = model(im)
            
            with autocast():
                pred = model(im)
                loss = loss_function(pred, label)

            p = F.softmax(pred, dim=1)

            test_loss += loss.item()
            correct += (p.argmax(1) == label.argmax(1)).type(torch.float).sum().item()
            
            y_true, y_pred = label.argmax(1).cpu().detach().numpy() , p.argmax(1).cpu().detach().numpy() 
            y_true_list = y_true_list + list(y_true)
            y_pred_list = y_pred_list + list(y_pred)
            
            # precision += precision_score( y_true , y_pred , average="macro" , zero_division=0 )
            # recall    += recall_score( y_true , y_pred , average="macro" , zero_division=0)
            # f1        += f1_score( y_true , y_pred , average="macro" , zero_division=0)
    
    precision = precision_score( y_true_list , y_pred_list , average="macro",zero_division=0)
    recall    = recall_score( y_true_list , y_pred_list , average="macro",zero_division=0)
    f1        = f1_score( y_true_list , y_pred_list , average="macro",zero_division=0)
    # print( len(y_true_list))
    # precision /= len(dataloader)
    # recall    /= len(dataloader)
    # f1        /= len(dataloader)
    
    test_loss /= len(dataloader)
    correct   /= size
    print("Valid  : loss:{:.5f}, correct:{:.5f}%, Precision:{:.5f} , Recall:{:.5f} , f1:{:.5f}".format( test_loss , correct*100 , precision , recall , f1))
    return test_loss , correct
