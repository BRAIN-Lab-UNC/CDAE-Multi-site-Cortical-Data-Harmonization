#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  19 17:13:04 2022

@author: Fenqiang Zhao

@contact: zhaofenqiang0221@gmail.com
"""

import os
import numpy as np
import torch
import time
import shutil

from s3pipe.models.models import Nonscanner_encoder, Scanner_encoder, Recon_decoder, MultiSiteClassifier_nonscanner
from s3pipe.surface.atlas import HarmonizationSphere
from utils_for_harmonization import split_files, split_files_jiale, split_files_jiale_rh
from torch.utils.tensorboard import SummaryWriter


###############################################################################
""" hyper-parameters, pay full attention! """

model_name = 'dcae'    #  'ae',  'dae', 'dcae'
learning_rate = 0.0005

weight_cycle = 0.5
weight_recon_latent = 4.0
weight_scanner_cls = 0.9
weight_scanner_cls1 = 0.5
weight_scanner_cls2 = 0.4
weight_fake_scanner_cls = 0.7
weight_corr = 10.0
weight_dist_corr = 3.0

# weight_cycle = 0.5
# weight_recon_latent = 4.0
# weight_scanner_cls = 0
# weight_scanner_cls1 = 0
# weight_scanner_cls2 = 0
# weight_fake_scanner_cls = 0
# weight_corr = 10.0
# weight_dist_corr = 3.0

# load = '20250919_test_rh_2nd'
load = '20250920_test_rh'

# experiment = "test_load"
# experiment = "test_rh"
# experiment = "test_rh_load"
# experiment = "test_rh_noload"
# experiment = "test_lh_3nd"
experiment = "test_rh_2nd"
# experiment = 'test_lh_noload'
# experiment = 'test_lh_noload_noshuffle'
# experiment = 'test_rh_noload_noshuffle'
    
experiment_path = time.strftime("%Y%m%d", time.localtime()) + '_' + experiment

save_path = os.path.join("./trained_models", experiment_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)
shutil.copy('./train_journal_jiale.py', os.path.join(save_path, 'train_journal_jiale.py'))

config_path = os.path.join("./tensorboard_config", experiment_path)
if not os.path.exists(config_path):
    os.makedirs(config_path)
writer = SummaryWriter(config_path)

###############################################################################
# os.system('rm -rf /media/ychenp/MyBook/Harmonization/scripts/log/' + model_name)
# writer = SummaryWriter('log/' + model_name)

n_ae_dim = 80
if model_name == 'ae':
    disentangle = False 
    cycle = False
if model_name == 'dae':
    disentangle = True 
    cycle = False
if model_name == 'dcae' :
    disentangle = True 
    cycle = True

device = torch.device('cuda:0')
n_res = 4
n_vertex = 40962
NUM_SCANNERS = 4
data_for_test=0.2

in_ch = 1
out_ch = 1
MAXAGE = 850


""" find files """
bcp_test_files, bcp_train_files, bcp_test_age, bcp_train_age, \
map_test_files, map_train_files, map_test_age, map_train_age, \
john_test_files, john_train_files, john_test_age, john_train_age, \
ndar_test_files, ndar_train_files, ndar_test_age, ndar_train_age, \
bcp_mean, bcp_std, map_mean, map_std, john_mean, john_std, ndar_mean, ndar_std = split_files_jiale_rh(data_for_test=data_for_test, MAXAGE=MAXAGE)

train_files = bcp_train_files + map_train_files + john_train_files + ndar_train_files
train_scanner_ids = list(np.zeros(len(bcp_train_files), dtype=np.int32)) + \
                    list(np.zeros(len(map_train_files), dtype=np.int32)+1) + \
                    list(np.zeros(len(john_train_files), dtype=np.int32)+2) + \
                    list(np.zeros(len(ndar_train_files), dtype=np.int32)+3)
train_age =  bcp_train_age + map_train_age + john_train_age + ndar_train_age
test_files = bcp_test_files + map_test_files + john_test_files + ndar_test_files
test_scanner_ids = list(np.zeros(len(bcp_test_files), dtype=np.int32)) + \
                    list(np.zeros(len(map_test_files), dtype=np.int32)+1) + \
                    list(np.zeros(len(john_test_files), dtype=np.int32)+2) + \
                    list(np.zeros(len(ndar_test_files), dtype=np.int32)+3)
test_age =  bcp_test_age + map_test_age + john_test_age + ndar_test_age


train_dataset = HarmonizationSphere(train_files, train_scanner_ids, train_age, n_vertex,
                                    bcp_mean, bcp_std, map_mean, map_std, john_mean, john_std, ndar_mean, ndar_std)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
test_dataset = HarmonizationSphere(test_files, test_scanner_ids, test_age, n_vertex,
                                    bcp_mean, bcp_std, map_mean, map_std, john_mean, john_std, ndar_mean, ndar_std)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

###############################################################################
"""Initialize models """
nonscanner_enc = Nonscanner_encoder(in_ch=1, level=7, n_res=4, rotated=0, complex_chs=32)
print("nonscanner_enc has {} paramerters in total".format(sum(x.numel() for x in nonscanner_enc.parameters())))
nonscanner_enc.to(device)
# nonscanner_enc.load_state_dict(torch.load('./trained_models/nonscanner_enc.'+ model_name +'.mdl'), strict=True)
nonscanner_enc.load_state_dict(torch.load('./trained_models/%s/nonscanner_enc.'%load+ model_name +'.mdl'), strict=True)

scanner_enc = Scanner_encoder(in_ch=1, level=7, n_res=6, rotated=0, complex_chs=8, n_scanners=NUM_SCANNERS)
print("scanner_enc has {} paramerters in total".format(sum(x.numel() for x in scanner_enc.parameters())))
scanner_enc.to(device)
# scanner_enc.load_state_dict(torch.load('./trained_models/scanner_enc.'+ model_name +'.mdl'), strict=True)
scanner_enc.load_state_dict(torch.load('./trained_models/%s/scanner_enc.'%load+ model_name +'.mdl'), strict=True)

recon_dec = Recon_decoder(in_ch=1, level=7, n_res=4, rotated=0, complex_chs=32, n_scanners=NUM_SCANNERS)
print("recon_dec has {} paramerters in total".format(sum(x.numel() for x in recon_dec.parameters())))
recon_dec.to(device)
# recon_dec.load_state_dict(torch.load('./trained_models/recon_dec.'+ model_name +'.mdl'), strict=True)
recon_dec.load_state_dict(torch.load('./trained_models/%s/recon_dec.'%load+ model_name +'.mdl'), strict=True)

model_c = MultiSiteClassifier_nonscanner(n_scanners=4, n_dim=256, level=4, n_res=2)
print("model_c has {} paramerters in total".format(sum(x.numel() for x in model_c.parameters())))
model_c.to(device)
# model_c.load_state_dict(torch.load('./trained_models/model_c.'+ model_name +'.mdl'), strict=True)
model_c.load_state_dict(torch.load('./trained_models/%s/model_c.'%load+ model_name +'.mdl'), strict=True)

optimizer_nonscanner_enc = torch.optim.Adam(nonscanner_enc.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_scanner_enc = torch.optim.Adam(scanner_enc.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_recon_dec = torch.optim.Adam(recon_dec.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizers = [optimizer_nonscanner_enc, optimizer_scanner_enc, optimizer_recon_dec]
optimizer_model_c = torch.optim.Adam(model_c.parameters(), lr=learning_rate, betas=(0.5, 0.999))

total_files = len(bcp_train_files)+len(map_train_files)+len(john_train_files)+len(ndar_train_files)
scanner_cls_weight = [total_files/len(bcp_train_files),
                      total_files/len(map_train_files),
                      total_files/len(john_train_files),
                      total_files/len(ndar_train_files)]
scanner_cls_weight = torch.tensor(scanner_cls_weight, device=device, dtype=torch.float32)

def get_learning_rate(epoch):
    if epoch < 5:
        lr = learning_rate * 1.0
    elif epoch < 10:
        lr = learning_rate * 0.5
    elif epoch < 20:
        lr = learning_rate * 0.2
    elif epoch < 30:
        lr = learning_rate * 0.1
    elif epoch < 50:
        lr = learning_rate * 0.05
    else:
        lr = learning_rate * 0.01
    return lr

def val_scanner_cls_during_training(train_dataloader):
    nonscanner_enc.eval()
    scanner_enc.eval()
    recon_dec.eval()
    
    total_correct_count = 0
    total_correct_count2 = 0
    
    total_correct_count_0 = 0
    total_count_0 = 0
    total_correct_count_1 = 0
    total_count_1 = 0
    total_correct_count_2 = 0
    total_count_2 = 0
    total_correct_count_3 = 0
    total_count_3 = 0
    
    total_correct_fake_count = 0
    
    for batch_idx, (data, scanner_id, file, age) in enumerate(train_dataloader):
        # print(file[0])
        real_data = data.squeeze(0).to(device)   # 40962*1
        real_target = scanner_id.squeeze().to(device)
        
        pred_real_prob, _ = scanner_enc(real_data)
        nonscanner_feat = nonscanner_enc(real_data)
        cls_non_scanner_feat, _ = model_c(nonscanner_feat)
        
        pred_scanner = pred_real_prob.squeeze().max(0)[1]
        pred_scanner2 = cls_non_scanner_feat.squeeze().max(0)[1]
        
        total_correct_count += pred_scanner == real_target
        total_correct_count2 += pred_scanner2 == real_target
        if real_target.item() == 0:
            total_correct_count_0 += pred_scanner == real_target
            total_count_0 += 1
        elif real_target.item() == 1:
            total_correct_count_1 += pred_scanner == real_target
            total_count_1 += 1
        elif real_target.item() == 2:
            total_correct_count_2 += pred_scanner == real_target
            total_count_2 += 1
        elif real_target.item() == 3:
            total_correct_count_3 += pred_scanner == real_target
            total_count_3 += 1
        else:
            raise NotImplementedError()
            
        # check if hamonizaed data can be correctly clssified
        fake_target = torch.tensor(0, device=device, dtype=torch.int64) 
        fake_prob = torch.zeros(NUM_SCANNERS, dtype=torch.float32, device=device)
        fake_prob[fake_target] = 1
        
        recon_fake_data = recon_dec(nonscanner_feat, fake_prob.unsqueeze(0))
        pred_fake_prob, _ = scanner_enc(recon_fake_data)
        pred_scanner = pred_fake_prob.squeeze().max(0)[1]
        total_correct_fake_count += pred_scanner == fake_target
        
    return total_correct_count/len(train_dataloader), \
            total_correct_count2/len(train_dataloader), \
            total_correct_count_0/total_count_0, \
            total_correct_count_1/total_count_1, \
            total_correct_count_2/total_count_2, \
            total_correct_count_3/total_count_3, \
            total_correct_fake_count/len(train_dataloader)
                       
#    dataiter = iter(train_dataloader)
#    data, scanner_id, file, age = dataiter.next()

for epoch in range(50):
    # with fake acc
    train_acc_scanner, train_acc_non_scanner, train_acc_0, train_acc_1, \
        train_acc_2, train_acc_3, train_fake_acc = val_scanner_cls_during_training(train_dataloader)
    print("train_acc_scanner, train_acc_non_scanner, train_acc_0, train_acc_1, train_acc_2, train_acc_3, train_fake_acc: ",
            train_acc_scanner.item(), train_acc_non_scanner.item(), 
            train_acc_0.item(), train_acc_1.item(), train_acc_2.item(), 
            train_acc_3.item(), train_fake_acc.item())
    test_acc_scanner, test_acc_non_scanner, test_acc_0, test_acc_1, \
        test_acc_2, test_acc_3, test_fake_acc = val_scanner_cls_during_training(test_dataloader)
    print("test_acc_scanner, test_acc_non_scanner, test_acc_0, test_acc_1, test_acc_2, test_acc_3, test_fake_acc: ", 
          test_acc_scanner.item(), test_acc_non_scanner.item(), 
          test_acc_0.item(), test_acc_1.item(), test_acc_2.item(), 
          test_acc_3.item(), test_fake_acc.item())
    # writer.add_scalars('data/train_Acc', {'train_acc_scanner': train_acc_scanner,
    #                                       'train_acc_non_scanner': train_acc_non_scanner,
    #                                       'train_acc_0': train_acc_0,
    #                                       'train_acc_1': train_acc_1,
    #                                       'train_acc_2': train_acc_2,
    #                                       'train_acc_3': train_acc_3,
    #                                       'train_fake_acc': train_fake_acc}, epoch)
    # writer.add_scalars('data/test_Acc', {'test_acc_scanner': test_acc_scanner,
    #                                       'test_acc_non_scanner': test_acc_non_scanner,
    #                                       'test_acc_0': test_acc_0,
    #                                       'test_acc_1': test_acc_1,
    #                                       'test_acc_2': test_acc_2,
    #                                       'test_acc_3': test_acc_3,
    #                                       'test_fake_acc': test_fake_acc}, epoch)
    writer.add_scalar('Train/acc_scanner', train_acc_scanner, epoch)
    writer.add_scalar('Train/acc_non_scanner', train_acc_non_scanner, epoch)
    writer.add_scalar('Train/acc_0', train_acc_0, epoch)
    writer.add_scalar('Train/acc_1', train_acc_1, epoch)
    writer.add_scalar('Train/acc_2', train_acc_2, epoch)
    writer.add_scalar('Train/acc_3', train_acc_3, epoch)
    writer.add_scalar('Train/fake_acc', train_fake_acc, epoch)

    writer.add_scalar('Test/acc_scanner', test_acc_scanner, epoch)
    writer.add_scalar('Test/acc_non_scanner', test_acc_non_scanner, epoch)
    writer.add_scalar('Test/acc_0', test_acc_0, epoch)
    writer.add_scalar('Test/acc_1', test_acc_1, epoch)
    writer.add_scalar('Test/acc_2', test_acc_2, epoch)
    writer.add_scalar('Test/acc_3', test_acc_3, epoch)
    writer.add_scalar('Test/fake_acc', test_fake_acc, epoch)

    lr = get_learning_rate(epoch)
    for optimizer in optimizers:
        optimizer.param_groups[0]['lr'] = lr
    print("learning rate = {}".format(lr))
    
    nonscanner_enc.train()
    scanner_enc.train()
    recon_dec.train()
    model_c.train()
    for batch_idx, (data, scanner_id, file, age) in enumerate(train_dataloader):
        # print(file[0])
        real_data = data.squeeze(0).to(device)   # 40962*1
        real_target = scanner_id.squeeze().to(device)
        real_prob = torch.zeros(NUM_SCANNERS, dtype=torch.float32, device=device)
        real_prob[real_target] = 1
        
        real_data_distance = real_data * real_data
        
        """  extract subject and scanner feat for real data """
        for optimizer in optimizers:
            optimizer.zero_grad()
        
        nonscanner_feat = nonscanner_enc(real_data)
        pred_real_prob, _ = scanner_enc(real_data)
        recon_real_data = recon_dec(nonscanner_feat, real_prob.unsqueeze(0))
        
        # if probability returned by Softmax, then treat it as a multiclass classification problem
        # loss_scanner_cls = torch.mean((real_prob - pred_real_prob.squeeze()) ** 2 * scanner_cls_weight)
        # if probability returned by Sigmoid, then first treat it as a single class problem then combine all classes
        loss_scanner_cls = torch.nn.functional.binary_cross_entropy(pred_real_prob.squeeze(), 
                                                                    real_prob, reduction='mean',
                                                                    weight=scanner_cls_weight)
        
        loss_recon_input = torch.mean(torch.abs(recon_real_data - real_data)) + \
                           torch.mean(torch.square(recon_real_data - real_data)) * 3.0
        # loss_mean = torch.abs(torch.mean(recon_real_data) - torch.mean(real_data))
        
        loss_real = loss_scanner_cls * weight_scanner_cls + loss_recon_input
        loss_real.backward()
        for optimizer in optimizers:
            optimizer.step()
        
        if disentangle:
            """  train non-scanner feature classification net """
            optimizer_model_c.zero_grad()
            nonscanner_feat = nonscanner_enc(real_data)
            cls_non_scanner_prob, _ = model_c(nonscanner_feat)
            
            # if probability returned by Sigmoid, then first treat it as a single class problem then combine all classes
            # loss_non_scanner_cls1 = torch.nn.functional.cross_entropy(cls_non_scanner_prob.squeeze(), 
            #                                                           real_target, weight=scanner_cls_weight)
            loss_non_scanner_cls1 = torch.nn.functional.binary_cross_entropy(cls_non_scanner_prob.squeeze(), 
                                                                             real_prob, reduction='mean',
                                                                             weight=scanner_cls_weight)
            loss_non_scanner_cls1 *= weight_scanner_cls1
            loss_non_scanner_cls1.backward()
            optimizer_model_c.step()
                    
            """  adversarialy train encoder using non-scanner feature """
            optimizer_nonscanner_enc.zero_grad()
            nonscanner_feat = nonscanner_enc(real_data)
            cls_non_scanner_prob2, _ = model_c(nonscanner_feat)
            
            # if probability returned by Sigmoid, then first treat it as a single class problem then combine all classes
            # gt_fake_prob = torch.empty_like(real_prob).fill_(1/NUM_SCANNERS)
            # loss_non_scanner_cls2 = torch.mean((gt_fake_prob - softmax(cls_non_scanner_prob2).squeeze()) ** 2 * scanner_cls_weight)
            gt_fake_prob = torch.empty_like(real_prob).fill_(0.5)
            loss_non_scanner_cls2 = torch.nn.functional.binary_cross_entropy(cls_non_scanner_prob2.squeeze(), 
                                                                              gt_fake_prob, reduction='mean',
                                                                              weight=scanner_cls_weight)
            loss_non_scanner_cls2 *= weight_scanner_cls2
            loss_non_scanner_cls2.backward()
            optimizer_nonscanner_enc.step()
        else:
            loss_non_scanner_cls1 = torch.tensor(0)
            loss_non_scanner_cls2 = torch.tensor(0)

        """ train fake recon and cycle consistentcy  """
        if cycle and real_target != 0:
            optimizer_recon_dec.zero_grad()
            optimizer_nonscanner_enc.zero_grad()
            
            # fake_target = torch.randint(0, NUM_SCANNERS, (1,), device=device).squeeze()
            # while fake_target == real_target:
            #     fake_target = torch.randint(0, NUM_SCANNERS, (1,), device=device).squeeze()
            fake_target = torch.tensor(0, device=device, dtype=torch.int64)
            fake_prob = torch.zeros(NUM_SCANNERS, dtype=torch.float32, device=device)
            fake_prob[fake_target] = 1
            
            nonscanner_feat = nonscanner_enc(real_data)
            recon_fake_data = recon_dec(nonscanner_feat, fake_prob.unsqueeze(0))
            nonscanner_feat_fake = nonscanner_enc(recon_fake_data)
            pred_fake_prob, _ = scanner_enc(recon_fake_data)
            
            recon_fake_data_distance = recon_fake_data * recon_fake_data
            """  loss for fake data """
            # if probability returned by Sigmoid, then first treat it as a single class problem then combine all classes
            # loss_fake_scanner_cls = torch.mean((fake_prob - cls_prob_fake.squeeze()) ** 2 * scanner_cls_weight)
            loss_fake_scanner_cls = torch.nn.functional.binary_cross_entropy(pred_fake_prob.squeeze(), 
                                                                              fake_prob, reduction='mean',
                                                                              weight=scanner_cls_weight)
            loss_recon_latent = torch.mean(torch.abs(nonscanner_feat - nonscanner_feat_fake)) + \
                                torch.mean(torch.square(nonscanner_feat - nonscanner_feat_fake)) * 10.0
                             
            loss_corr = 1 - ((recon_fake_data - recon_fake_data.mean()) * (real_data - real_data.mean())).mean() \
                        / recon_fake_data.std() / real_data.std()
                        
            if real_data_distance.std() < 1e-12:
                raise NotImplementedError('real_data_distance.std() < 1e-12 error')
            loss_dist_corr = 1 - ((real_data_distance - real_data_distance.mean()) *\
                (recon_fake_data_distance - recon_fake_data_distance.mean())).mean() \
                    / real_data_distance.std() / recon_fake_data_distance.std()
    
            recon_real_data_from_fake = recon_dec(nonscanner_feat_fake, real_prob.unsqueeze(0))
            loss_cycle =  torch.mean(torch.abs(recon_real_data_from_fake - real_data)) + \
                          torch.mean(torch.square(recon_real_data_from_fake - real_data)) * 3.0
                          
            loss_fake = loss_fake_scanner_cls * weight_fake_scanner_cls + \
                        loss_recon_latent * weight_recon_latent + \
                        loss_corr * weight_corr + \
                        loss_cycle * weight_cycle + \
                        loss_dist_corr * weight_dist_corr

            loss_fake.backward()
            optimizer_recon_dec.step()
            optimizer_nonscanner_enc.step()
        else:
            loss_fake_scanner_cls = torch.tensor(0) 
            loss_recon_latent = torch.tensor(0)
            loss_cycle = torch.tensor(0)
            loss_corr = torch.tensor(0)
            loss_fake = torch.tensor(0)
            loss_dist_corr = torch.tensor(0)
      
        print("[{}:{}/{}] loss_recon_input={:5.4f} loss_scanner_cls={:5.4f} loss_non_scanner_cls1={:5.4f} loss_non_scanner_cls2={:5.4f} loss_fake_scanner_cls={:5.4f} loss_recon_latent={:5.4f} loss_corr={:5.4f} loss_cycle={:5.4f}".format(epoch,
              batch_idx, len(train_dataloader),
              loss_recon_input.item(), loss_scanner_cls.item(), loss_non_scanner_cls1.item(), 
              loss_non_scanner_cls2.item(), loss_fake_scanner_cls.item(), loss_recon_latent.item(),
              loss_corr.item(), loss_cycle.item()))
        
        # writer.add_scalars('Train/loss',
        #                     {'loss_recon_input': loss_recon_input.item(),
        #                      'loss_scanner_cls': loss_scanner_cls.item()*weight_scanner_cls,
        #                      'loss_non_scanner_cls1': loss_non_scanner_cls1.item(),
        #                      'loss_non_scanner_cls2': loss_non_scanner_cls2.item(),
        #                      'loss_fake_scanner_cls': loss_fake_scanner_cls.item()*weight_fake_scanner_cls,
        #                      'loss_recon_latent': loss_recon_latent.item()*weight_recon_latent,
        #                      'loss_corr': loss_corr.item()*weight_corr,
        #                      'loss_cycle': loss_cycle.item()*weight_cycle,
        #                      'loss_dist_corr': loss_dist_corr.item()* weight_dist_corr
        #                      },
        #                      epoch*len(train_dataloader)+batch_idx)
                            # 'loss_mean': loss_mean.item()*weight_mean

        writer.add_scalar('Train/loss_recon_input', loss_recon_input.item(), epoch*len(train_dataloader)+batch_idx)
        writer.add_scalar('Train/loss_scanner_cls', loss_scanner_cls.item()*weight_scanner_cls, epoch*len(train_dataloader)+batch_idx)
        writer.add_scalar('Train/loss_non_scanner_cls1', loss_non_scanner_cls1.item(), epoch*len(train_dataloader)+batch_idx)
        writer.add_scalar('Train/loss_non_scanner_cls2', loss_non_scanner_cls2.item(), epoch*len(train_dataloader)+batch_idx)
        writer.add_scalar('Train/loss_fake_scanner_cls', loss_fake_scanner_cls.item()*weight_fake_scanner_cls, epoch*len(train_dataloader)+batch_idx)
        writer.add_scalar('Train/loss_recon_latent', loss_recon_latent.item()*weight_recon_latent, epoch*len(train_dataloader)+batch_idx)
        writer.add_scalar('Train/loss_corr', loss_corr.item()*weight_corr, epoch*len(train_dataloader)+batch_idx)
        writer.add_scalar('Train/loss_cycle', loss_cycle.item()*weight_cycle, epoch*len(train_dataloader)+batch_idx)
        writer.add_scalar('Train/loss_dist_corr', loss_dist_corr.item()* weight_dist_corr, epoch*len(train_dataloader)+batch_idx)

    # torch.save(nonscanner_enc.state_dict(), '/media/ychenp/MyBook/Harmonization/scripts/trained_models/nonscanner_enc.'+ model_name +'.mdl')
    # torch.save(scanner_enc.state_dict(), '/media/ychenp/MyBook/Harmonization/scripts/trained_models/scanner_enc.'+ model_name +'.mdl')
    # torch.save(recon_dec.state_dict(), '/media/ychenp/MyBook/Harmonization/scripts/trained_models/recon_dec.'+ model_name +'.mdl')
    # torch.save(model_c.state_dict(), '/media/ychenp/MyBook/Harmonization/scripts/trained_models/model_c.'+ model_name +'.mdl')

    torch.save(nonscanner_enc.state_dict(), os.path.join(save_path, 'nonscanner_enc.'+ model_name +'.mdl'))
    torch.save(scanner_enc.state_dict(), os.path.join(save_path, 'scanner_enc.'+ model_name +'.mdl'))
    torch.save(recon_dec.state_dict(), os.path.join(save_path, 'recon_dec.'+ model_name +'.mdl'))
    torch.save(model_c.state_dict(), os.path.join(save_path, 'model_c.'+ model_name +'.mdl'))
