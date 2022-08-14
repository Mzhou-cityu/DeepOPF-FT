import torch
import numpy as np
import math
# This file is used to set all the global varibles for the programe
global bus, gen, gencost, branch, baseMVA, bus_slack
global idxPg, idxQg, bus_Pg, bus_Qg, bus_PQg, PG_Lowbound, PG_Upbound, QG_Lowbound, QG_Upbound
global input_channels, output_channels_vm, output_channels_va, hidden_units
global Xtest, Rtest, idxPd, idx_Qd

global training_loader_vm, training_loader_va, test_loader_vm, test_loader_va
global hisVm_min, hisVm_max
global Pdtest, Qdtest, Pd_train, Qd_train, Pgtest, Qgtest
global Real_Va, Real_Vm
global Real_Pg, Real_Qg, Real_Pd, Real_Qd, Real_Vmtrain, Real_Vatrain
global VmLb, VmUb, Real_V, Real_Pg
global VA_Upbound, VA_Lowbound
global system
global input_channels, output_channels, hidden_units, hisVm_max, hisVm_min
VaUb = math.pi
VaLb = -math.pi

## whether there is GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print("Let's use", torch.cuda.device_count(), "GPUs!")

Nbus = 6  # number of buses

# **********************  DNN hyper parameters  **************************
Epoch = 100  # maximum epoch for training
Epoch_pre = 150  # maximum epoch of re-traiing
p_epoch = 10   # output the training loss every p_epoch
batch_size_training = 128  # mini-batch size for training
batch_size_test = 512  # mini-batch size for test
Lr = 1e-5 # learning rate
gamma = 1  # discount factor for the learning rate
k_dV = 1  # coefficient for dVa & dVm in post-processing
DELTA = 1e-4  # threshold of violation
scale_vm = torch.tensor([1]).float()  # scaling of output Vm
scale_va = torch.tensor([1]).float()  # scaling of output Va
NN_size = 128
mse_weight = 1.0
penalty_weight = 1.0
# hidden layers for voltage magnitude (Vm) prediction
khidden = np.array([8, 4, 2], dtype=int)
# name of hidden layers for result saving
nmDNN = 'DNN'
for i in range(khidden.shape[0]):
    nmDNN = nmDNN + str(khidden[i])
LDNN = khidden.shape[0]  # number of hidden layers

# **********************  DNN model train configurations  **************************
pretrain_flag = 0   #  1: load pre-trained model to re-train, 0 train from scratch
mode = '_EM'   # EM for embedded training
#mode = '_DISV1'   #DISV1 for discrete training version 1
#mode = '_DISV2'   #DISV2 for discrete training version 2

# **********************  DNN model test configurations  **************************
test_flag = 1 # 1: test mode 0; train mode
flag_hisv = 1  # 1-use historical V to calculate dV;0-use predicted V to calculate dV
va_test_flag = 1    # 1: use real va to replace predicted va
vm_test_flag = 1   # 1: use real vm to replace predicted vm
train_test_flag = 1  # 1:use training data as the test data; 0 use test dataset
Nsystem = 2

#******************** train-test data configuration ***********************
Ntrain = 100
Ntest = 100
if(train_test_flag == 1):
    Ntest = Ntrain

R_UpBound = 100000000

#******************** path for model files ***********************
PATH = './model/model' + str(Epoch) +'.pth'
PATH_pre = './model/model_pre' + str(Epoch_pre) + '.pth'
# columns in the excel output
Test_case = ['Flexible topology', 'Flexible admittance']
# rows in the excel output
Evaluation_index = ['opt', 'Va_ratio', 'Vm_ratio', 'Pg_ratio', 'Pg_degree',
                    'Qg_ratio', 'Qg_degree', 'Branch_ratio', 'Pd_ratio', 'Qdratio']
data_path = './data'
result_path = './result'
