from MyPackage.Net import Net
import MyPackage.config as config
import torch
import numpy as np
from MyPackage.evaluation_functions import *
from MyPackage.get_violation import *
from MyPackage.get_dV import get_dV
def test_model():
    Pred_Va, Pred_Vm_clip, Pred_V = get_Pred_V()
    print("**************** before post processing *******************")
    PG_violation_gen, QG_violation_gen, PG_violation, QG_violation, PQ_violation_num, PQG_violation_index = \
        get_performance(Pred_Va, Pred_Vm_clip, Pred_V)   # performance before post processing
    Pred_Va_post, Pred_Vm_post_clip, Pred_V_post = post_processing(Pred_Va, Pred_Vm_clip, Pred_V, PG_violation_gen, QG_violation_gen,
                                                                   PG_violation, QG_violation, PQ_violation_num, PQG_violation_index)
    print("*******************After post processing**********************")
    get_performance(Pred_Va_post, Pred_Vm_post_clip, Pred_V_post)  # performance after post processing

def get_va_prediction():
    # ********  load trained model  ************
    if (config.pretrain_flag == 1):
        Path = config.PATH_pre
    else:
        Path = config.PATH
    # load trained model
    print('load trained model: model_load')
    model = Net(config.input_channels, config.output_channels, config.hidden_units, config.khidden)
    model.load_state_dict(torch.load(Path, map_location = config.device))
    model.eval()
    model.to(config.device)

    VA_prediction = torch.zeros((config.Ntest, config.Nbus - 1))
    for step, (test_x, test_y) in enumerate(config.test_loader):
        test_x = test_x.to(config.device)
        VA_prediction[step] = model(test_x)[:, 0:config.Nbus-1]

    VA_prediction = VA_prediction.cpu()
    VA_prediction = VA_prediction.detach() / config.scale_va

    # Va with slack bus
    Pred_Va = VA_prediction.clone().numpy()
    Pred_Va = np.insert(Pred_Va, config.bus_slack, values=0, axis=1)

    if (config.va_test_flag == 1):
        Pred_Va = config.Real_Va
    return Pred_Va

def get_vm_prediction():
    # ********  load trained model  ************

    Path = config.PATH
    print('load trained model')
    model = Net(config.input_channels, config.output_channels, config.hidden_units, config.khidden)
    model.load_state_dict(torch.load(Path, map_location=config.device))
    model.eval()
    model.to(config.device)

    # ***********  Predicted data  ******************
    VM_prediction = torch.zeros((config.Ntest, config.Nbus))
    for step, (test_x, test_y) in enumerate(config.test_loader):
        test_x = test_x.to(config.device)
        VM_prediction[step] = model(test_x)[:, config.Nbus-1:2*config.Nbus-1]
    VM_prediction = VM_prediction.cpu()
    Pred_Vm = VM_prediction.detach() / config.scale_vm * (config.VmUb - config.VmLb) + config.VmLb
    Pred_Vm_clip = get_clamp(Pred_Vm, config.hisVm_min, config.hisVm_max)

    # *********** use test vm as vm preidtion ***********
    if (config.vm_test_flag == 1):
        Pred_Vm_clip = config.Real_Vm
    return Pred_Vm_clip

def get_Pred_V():
    Pred_Va = get_va_prediction()
    Pred_Vm_clip = get_vm_prediction()
    Pred_V = Pred_Vm_clip.clone().numpy() * np.exp(1j * Pred_Va)  # predicted V
    print("******************* VA VM *********************************")
    mae_Vmtest = get_abs_error(config.Real_Vm, Pred_Vm_clip.detach())
    mae_Vatest = get_abs_error(torch.from_numpy(config.Real_Va), torch.from_numpy(Pred_Va))
    Real_va_without_slack = np.delete(config.Real_Va, config.bus_slack, axis=1)
    Pred_va_without_slack = np.delete(Pred_Va, config.bus_slack, axis=1)
    print('the mean average abosolute error of VM is: {name}  '.format(name=(mae_Vmtest)))
    print('the mean average abosolute error of VA is: {name} '.format(name=(mae_Vatest)))
    return Pred_Va, Pred_Vm_clip, Pred_V

def get_performance(Pred_Va, Pred_Vm_clip, Pred_V):

    #  ************   Pg Qg prediction  *******************
    Pred_Pg, Pred_Qg, Pred_Pd, Pred_Qd = get_genload(Pred_V, config.Pdtest, config.Qdtest, config.bus_Pg, config.bus_Qg)



    #  ************   Pg Qg violation  *******************
    PG_violation_ratio, QG_violation_ratio, PG_violation_gen, QG_violation_gen, PG_violation, QG_violation, PQ_violation_num, PQG_violation_index = get_PQ_violation(Pred_Pg, Pred_Qg)

    # ***********  branch violation *******************
    branch_violation = get_branch_violation(Pred_V)

    # ******************* V theta violation ***************************
    VM_violation_ratio, VA_violation_ratio, VM_violation_bus, \
    VA_violation_bus = get_V_violation(Pred_Vm_clip, Pred_Va)

    # load satisfaction
    mre_Pd = get_load_mismatch_rate(torch.from_numpy(config.Real_Pd.sum(axis=1)),
                                    torch.from_numpy(Pred_Pd.sum(axis=1)))

    mre_Qd = get_load_mismatch_rate(torch.from_numpy(config.Real_Qd.sum(axis=1)),
                                    torch.from_numpy(Pred_Qd.sum(axis=1)))

    # optimality loss
    Pred_cost = get_Pgcost(Pred_Pg, config.idxPg, config.gencost)

    Real_cost = get_Pgcost(config.Real_Pg, config.idxPg, config.gencost)
    opt_gap = np.mean(np.divide((Pred_cost - Real_cost), Real_cost)) * 100

    print("The PG satisfaction rate is: {name}%".format(name=100 - PG_violation_ratio))
    print("The QG satisfaction rate is: {name}%".format(name=100 - QG_violation_ratio))
    print("The Branch satisfaction rate is: {name}%".format(name=100 - branch_violation))
    print("The VM satisfaction rate is: {name}%".format(name=100 - VM_violation_ratio))
    print("The VA satisfaction rate is: {name}%".format(name=100 - VA_violation_ratio))
    print("The Pd satisfaction rate is: {name}%".format(name=100 - torch.mean(mre_Pd)))
    print("The Qd satisfaction rate is: {name}%".format(name=100 - torch.mean(mre_Qd)))
    print("The optimality gap is: {name}%".format(name=(opt_gap)))
    return PG_violation_gen, QG_violation_gen, PG_violation, QG_violation, PQ_violation_num, PQG_violation_index

def post_processing(Pred_Va, Pred_Vm_clip, Pred_V, PG_violation_gen, QG_violation_gen, PG_violation,
                    QG_violation, PQ_violation_num, PQG_violation_index):
    # post-processing of Vm Va
    Pred_Va_post = Pred_Va.copy()
    Pred_Vm_post = Pred_Vm_clip.clone().numpy()

    dV = get_dV(Pred_V, PG_violation_gen, QG_violation_gen, PG_violation, QG_violation, config.bus_Pg, config.bus_Qg,
                 PQ_violation_num)

    Pred_Va_post[PQG_violation_index, :] = Pred_Va[PQG_violation_index, :] - dV[:, 0:config.Nbus]   # revised va
    Pred_Va_post[:, config.bus_slack] = 0
    Pred_Vm_post[PQG_violation_index, :] = Pred_Vm_clip.numpy()[PQG_violation_index, :] - dV[:, config.Nbus:2 * config.Nbus]  # revised vm
    Pred_Vm_post_clip = get_clamp(torch.from_numpy(Pred_Vm_post), config.hisVm_min, config.hisVm_max)   # vm clip
    Pred_V_post = Pred_Vm_post_clip.numpy() * np.exp(1j * Pred_Va_post)  # revised V
    return Pred_Va_post, Pred_Vm_post_clip, Pred_V_post

