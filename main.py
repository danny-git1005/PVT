import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from timm.scheduler.cosine_lr import CosineLRScheduler
from torchsummary import summary
import torchvision
from torchvision import transforms
import os
import configparser
from DataClass import AllDataset
from PVT import PVT
from utils import train , valid , get_data, img_transform
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    
# Tranfromer = transforms.Compose([
#         transforms.Resize((224,224)),
#         # transforms.RandomVerticalFlip(),
#         # transforms.RandomCorp()
#         transforms.ToTensor(),
#         transforms.Normalize([0.439, 0.459, 0.406], [0.185, 0.186, 0.229]),
#     ])

    
if __name__ == "__main__":
    torch.manual_seed(torch.initial_seed())

    config = configparser.ConfigParser()
    # config.read('./config/config_data_city.ini')
    config.read('./config/config_data_bird.ini')
    
    train_dir = config['data']['train_dir']
    valid_dir = None
    test_dir  = None
    if config['data']['val_dir'] != 'None':
        valid_dir = config['data']['val_dir']
    if config['data']['test_dir'] != 'None':
        test_dir  = config['data']['test_dir']
    
    train_imgs , valid_imgs , test_imgs , classInt = get_data( train_dir=train_dir , val_dir=valid_dir , test_dir=test_dir )
    
    train_dataset = AllDataset( train_imgs , classInt , transforms=None )
    if valid_imgs and test_imgs:
        val_dataset   = AllDataset( valid_imgs , classInt , transforms=None )
        test_dataset  = AllDataset( test_imgs  , classInt , transforms=None )
    else:
        train_size = int(len(train_dataset)*0.8)
        val_size   = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # train_dataset = torchvision.datasets.ImageFolder(root='/data/practice_data/city/Images', target_transform=Tranfromer)
    # train_size = int(len(train_dataset)*0.8)
    # val_size   = len(train_dataset) - train_size
    # train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    # print(type(train_dataset))
    model_name = 'pvt small'        
    model = PVT(
        img_size    = config[model_name].getint('img_size'),
        patch_size  = config[model_name].getint('patch_size'),
        classes     = config[model_name].getint('classes'),
        embed_dim   = json.loads(config[model_name].get('embed_dim')),
        num_heads   = json.loads(config[model_name].get('num_heads')),
        mlp_ratio   = json.loads(config[model_name].get('mlp_ratio')), 
        qkv_bias    = False if config[model_name].get('qkv_bias') == 'False'else True,
        drop        = config[model_name].getfloat('drop'),
        attn_drop   = config[model_name].getfloat('attn_drop'),
        block_depth = json.loads(config[model_name].get('block_depth')),
        sr_ratio    = json.loads(config[model_name].get('sr_ratio')),
        num_stage   = config[model_name].getint('num_stage'),
    )

    summary( model, (3,224,224) )
    
    batch         = 144
    epoch         = 400
    learning_rate = 0.00005

    train_loader = DataLoader( train_dataset , batch_size=batch , shuffle=True , num_workers=12 , pin_memory=True , drop_last=True)
    val_loader   = DataLoader( val_dataset   , batch_size=batch  , shuffle=True , num_workers=12 , pin_memory=True , drop_last=True)

    loss      = nn.CrossEntropyLoss(reduction='mean' , label_smoothing=0)
    loss_test = nn.CrossEntropyLoss(reduction='mean')

    optimizer    = torch.optim.AdamW(model.parameters() , lr=learning_rate)
    lr_scheduler = CosineLRScheduler(
                        optimizer      = optimizer, 
                        t_initial      = epoch // 4,
                        warmup_t       = config['cosLrScheduler'].getint('warmup_t'), 
                        warmup_lr_init = config['cosLrScheduler'].getfloat('warmup_lr_init'),
                        cycle_decay    = config['cosLrScheduler'].getfloat('cycle_decay'),
                        cycle_limit    = config['cosLrScheduler'].getint('cycle_limit'),
                        k_decay        = config['cosLrScheduler'].getfloat('k_decay'),
                        lr_min         = config['cosLrScheduler'].getfloat('lr_min')
                    )
  
    writer = SummaryWriter( log_dir = '/code/tb_logs/Bird_MyPVT_'+ model_name.split(' ')[1] +
                                                  '_lr-' + str(learning_rate)+
                                                  '_batch-' + str(batch)+ 
                                                  '_label_smooth-0'+
                                                  '_CosineLRScheduler'+
                                                  '_accumlate-2_ep3')


    for i in range(epoch):
        print('epoch:{}/{}'.format(i,epoch))
        scaler = amp.GradScaler()
 
        train_loss , train_acc = train( dataloader=train_loader , model=model , loss_function=loss , optimizer=optimizer , lr_scheduler=lr_scheduler , scaler=scaler  )
        val_loss   , val_acc   = valid( dataloader=val_loader , model=model , loss_function=loss_test )


        writer.add_scalar( "Accuracy / Train" , train_acc  , i )
        writer.add_scalar( "Loss     / Train" , train_loss , i )
        writer.add_scalar( "Accuracy / val"   , val_acc    , i )
        writer.add_scalar( "Loss     / val"   , val_loss   , i )

        if val_acc > 0.8 :
            modelname = './model/model_' + str(i) + '.pt'
            torch.save(model.state_dict(), modelname)
            print('Save model {}'.format(i))
            print('\n')