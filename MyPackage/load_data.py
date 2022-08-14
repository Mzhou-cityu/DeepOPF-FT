import MyPackage.config as config
import scipy.io
import math
import numpy as np
import torch
import torch.utils.data as Data
from MyPackage.evaluation_functions import get_genload

def load_data():
    load_train_data()
    load_test_data()

#********************** load training data function*************
def load_train_data():
    global hisVm_min, hisVm_max

    mat_train = scipy.io.loadmat('./data/train_case' + str(config.Nbus) + config.mode + '.mat')
    # **********  load data from the dataset *****************************
    RPd0_train = mat_train['RPd_train'][0:config.Ntrain, :]  # Pd
    RQd0_train = mat_train['RQd_train'][0:config.Ntrain, :]  # Qd
    RPg_train = mat_train['RPg_train'][0:config.Ntrain, :]  # Pg
    RQg_train = mat_train['RQg_train'][0:config.Ntrain, :]  # Qg
    RX_train = mat_train['RX_train'][0:config.Ntrain, :]  # X
    RR_train = mat_train['RR_train'][0:config.Ntrain, :]  # R
    RN_train = mat_train['RN_train'][0:config.Ntrain, :]  # RN
    RD_train = mat_train['RD_train'][0:config.Ntrain, :]  # RD
    config.VmLb = mat_train['VmLb']  # lower bound of Vm
    config.VmUb = mat_train['VmUb']  # upper bound of Vm
    Va_train = mat_train['RVa_train'][0:config.Ntrain, :] * math.pi / 180
    Vm_train = mat_train['RVm_train'][0:config.Ntrain, :]


    load_idx = np.arange(0, config.Nbus, 1, int)
    # input data: only contain non-zeros loads
    config.idx_Pd = np.squeeze(np.where(np.abs(RPd0_train[0, :]) > 0), axis=0)
    config.idx_Qd = np.squeeze(np.where(np.abs(RQd0_train[0, :]) > 0), axis=0)

    # ********************* input data ***************************
    load_train = np.concatenate((RPd0_train[:, config.idx_Pd], RQd0_train[:, config.idx_Qd]), axis=1) / config.baseMVA

    dn_train = np.concatenate((RD_train, RN_train), axis=1)
    x_train = np.concatenate((load_train, dn_train), axis=1)

  #  x_train = load_train
    x_train_tensor = torch.from_numpy(x_train).float()

    #***********   output data   *********************************
    Va_train = np.delete(Va_train, config.bus_slack, axis=1)     # do not need to predict voltage angles of slack bus
    Va_train_normalization = (Va_train - config.VaLb) / (config.VaUb - config.VaLb)
    VM_train_normalization = (Vm_train - config.VmLb) / (config.VmUb - config.VmLb)  # scaled Vm
    #  convert training data into torch and do scaling
    VM_train_normalization = torch.from_numpy(VM_train_normalization).float()
    Va_train_normalization = torch.from_numpy(Va_train_normalization).float()

    # convert training data to tensor
    VM_train_tensor = VM_train_normalization.float()
    VA_train_tensor = Va_train_normalization.float()

    VAVM_train_tensor = torch.cat((VA_train_tensor, VM_train_tensor), dim = 1)

    # *********  batch data into a data loader ***************
    training_dataset = Data.TensorDataset(x_train_tensor, VAVM_train_tensor)
    config.training_loader = Data.DataLoader(
        dataset = training_dataset,
        batch_size = config.batch_size_training,
        shuffle = False,
    )
    '''
    training_dataset_va = Data.TensorDataset(x_train_tensor, VA_train_tensor)
    config.training_loader_va = Data.DataLoader(
        dataset = training_dataset_va,
        batch_size = config.batch_size_training,
        shuffle=True,
    )
    '''
    # ************ Pd Qd of samples--- for test the trained model on the training set ********
    RPd_train = np.zeros((config.Ntrain, config.Nbus))
    RQd_train = np.zeros((config.Ntrain, config.Nbus))
    RPd_train[:, load_idx] = RPd0_train
    RQd_train[:, load_idx] = RQd0_train
    config.Pdtrain = RPd_train / config.baseMVA
    config.Qdtrain = RQd_train / config.baseMVA

    # ***************** historical voltage----for post processing   ****************************
    config.Real_Vmtrain = VM_train_tensor / config.scale_vm * (config.VmUb - config.VmLb) + config.VmLb
    config.Real_Vatrain = VA_train_tensor * (config.VaUb - config.VaLb) + config.VaLb
    config.hisVm_max, _ = torch.max(config.Real_Vmtrain, dim=0)
    config.hisVm_min, _ = torch.min(config.Real_Vmtrain, dim=0)
    his_Va = np.mean(np.insert(config.Real_Vatrain.numpy(), config.bus_slack, values=0, axis=1), axis=0)
    his_Vm = np.mean(config.Real_Vmtrain.numpy(), axis=0)
    his_V = his_Vm * np.exp(1j * his_Va)  # historical V

    # *************   DNN structure  *******************************
    config.input_channels = x_train_tensor.shape[1]
    config.output_channels = VAVM_train_tensor.shape[1]
    config.hidden_units = config.NN_size

#********************** load test data function*************
def load_test_data():

    mat_test = scipy.io.loadmat('./data/test_case' + str(config.Nbus)  + config.mode + '.mat')
    #   load_idx = np.squeeze(mat['load_idx']).astype(int) - 1
    load_idx = np.arange(0, config.Nbus, 1, int)

    RPd0_test = mat_test['RPd_test'][0:config.Ntest, :]  # Pd
    RQd0_test = mat_test['RQd_test'][0:config.Ntest, :]  # Qd
    RPg_test = mat_test['RPg_test'][0:config.Ntest, :]  # Pg
    RQg_test = mat_test['RQg_test'][0:config.Ntest, :]  # Qg
    RX_test = mat_test['RX_test'][0:config.Ntest, :]  # X
    RR_test = mat_test['RR_test'][0:config.Ntest, :]  # R
    config.Ff_test = mat_test['RFf_test'][0:config.Ntest, :]  # Ff
    config.Ft_test = mat_test['RFt_test'][0:config.Ntest, :]  # Ff
    #    Ft_test = mat['RFt_test']  # Ft
    RN_test = mat_test['RN_test'][0:config.Ntest, :]  # RN
    RD_test = mat_test['RD_test'][0:config.Ntest, :]  # RD

    config.DN_test = abs(RD_test + RN_test)

    Va_test = mat_test['RVa_test'][0:config.Ntest, :] * math.pi / 180
    Vm_test = mat_test['RVm_test'][0:config.Ntest, :]
    config.f_test = mat_test['Rf_test'][0:config.Ntest, :]  # Ff

    # ********************** input data ************************************
    load_test = np.concatenate((RPd0_test[:, config.idx_Pd], RQd0_test[:, config.idx_Qd]), axis=1) / config.baseMVA

    dn_test = np.concatenate((RD_test, RN_test), axis=1)
    x_test = np.concatenate((load_test, dn_test), axis=1)

   # x_test = load_test
    # convert test data to tensor
    x_test_tensor = torch.from_numpy(x_test).float()

    # ********************** output data ************************************
    Va_test = np.delete(Va_test, config.bus_slack, axis=1) # do not need to predict voltaeg angles of slack bus
    Va_test_normalization = (Va_test - config.VaLb) / (config.VaUb - config.VaLb)
    VM_test_normalization = (Vm_test - config.VmLb) / (config.VmUb - config.VmLb)  # scaled Vm
    VM_test_normalization = torch.from_numpy(VM_test_normalization).float()
    Va_test_normalization = torch.from_numpy(Va_test_normalization).float()

    VM_test_tensor = VM_test_normalization.float()
    VA_test_tensor = Va_test_normalization.float()

    VAVM_test_tensor = torch.cat((VA_test_tensor, VM_test_tensor), dim = 1)

    # *********  batch data into a data loader ***************
    batch_size_test = 1
    test_dataset = Data.TensorDataset(x_test_tensor, VAVM_test_tensor)
    config.test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
    )
    # **********  test load generation   ********************
    RPd_test = np.zeros((config.Ntest, config.Nbus))
    RQd_test = np.zeros((config.Ntest, config.Nbus))
    RPd_test[:, load_idx] = RPd0_test
    RQd_test[:, load_idx] = RQd0_test
    config.Pdtest = RPd_test / config.baseMVA
    config.Qdtest = RQd_test / config.baseMVA
    config.Pgtest = RPg_test[:, config.idxPg] / config.baseMVA
    config.Qgtest = RQg_test[:, config.idxQg] / config.baseMVA
    config.Qgtest = config.Qgtest.squeeze()
    config.Xtest = RX_test
    config.Rtest = RR_test

    # *************   real Vm Va for testing samples   ******************
    config.Real_Vm = VM_test_tensor  * (config.VmUb - config.VmLb) + config.VmLb
    config.Real_Va = VA_test_tensor.clone().numpy() * (config.VaUb - config.VaLb) + config.VaLb   # Va with slack bus value
    config.Real_Va = np.insert(config.Real_Va, config.bus_slack, values=0, axis=1)  # Va with slack bus value
    config.Real_V = config.Real_Vm.numpy() * np.exp(1j * config.Real_Va)  # complex value: Vm + j* Va
    config.Real_Pg, config.Real_Qg, config.Real_Pd, config.Real_Qd = get_genload(
        config.Real_V, config.Pdtest, config.Qdtest, config.bus_Pg, config.bus_Qg)     # Pg QG

