import MyPackage.config as config
import numpy as np
from pypower.makeYbus import makeYbus
from pypower.idx_brch import F_BUS, T_BUS, BR_STATUS, PF, PT, QF, QT

from numpy import flatnonzero as find

## violation calculation function
# Pg Qg violation
def get_PQ_violation(Pred_Pg, Pred_Qg):

    #PG violation
    PG_violation_up = Pred_Pg - config.PG_Upbound
    PG_violation_up[PG_violation_up < config.DELTA] = 0

    PG_violation_low =  config.PG_Lowbound - Pred_Pg
    PG_violation_low[PG_violation_low < config.DELTA] = 0

    PG_violation = PG_violation_up + PG_violation_low

    PG_violation_ratio = np.count_nonzero(PG_violation) / (np.shape(PG_violation)[0] * np.shape(PG_violation)[1])
    PG_violation_gen = np.sum(PG_violation, axis = 1)

    # QG violation
    QG_violation_up = Pred_Qg - config.QG_Upbound
    QG_violation_up[QG_violation_up < config.DELTA] = 0

    QG_violation_low = config.QG_Lowbound - Pred_Qg
    QG_violation_low[QG_violation_low < config.DELTA] = 0

    QG_violation = QG_violation_up + QG_violation_low

    QG_violation_ratio = np.count_nonzero(QG_violation) / (np.shape(QG_violation)[0] * np.shape(QG_violation)[1])
    QG_violation_gen = np.sum(QG_violation, axis = 1)

    #***** indicator function for PQ violation
    PG_violation_up[PG_violation_up < 0] = 0
    PG_violation_up[PG_violation_up > 0] = 1
    PG_violation_low[PG_violation_low < 0] = 0
    PG_violation_low[PG_violation_low > 0] = 1
    QG_violation_up[QG_violation_up < 0] = 0
    QG_violation_up[QG_violation_up > 0] = 1
    QG_violation_low[QG_violation_low < 0] = 0
    QG_violation_low[QG_violation_low > 0] = 1
    PG_violation_indicator = PG_violation_up + PG_violation_low
    QG_violation_indicator  = QG_violation_up + QG_violation_low
    PQ_violation_indicator  = np.concatenate((PG_violation_indicator , QG_violation_indicator ), axis = 1)

    PQ_violation_gen = PG_violation_gen + QG_violation_gen
    PQ_violation_num = np.count_nonzero(PQ_violation_gen)
    PQG_violation_index = np.where(PQ_violation_gen > 0)

    return PG_violation_ratio, QG_violation_ratio, PG_violation_gen, QG_violation_gen, PG_violation, QG_violation, PQ_violation_num, PQG_violation_index


def get_PQ_violation_grad(Pred_Pg, Pred_Qg):
    # PG violation
    PG_violation_up = Pred_Pg - config.PG_Upbound
    PG_violation_up[PG_violation_up < config.DELTA] = 0

    PG_violation_low = config.PG_Lowbound - Pred_Pg
    PG_violation_low[PG_violation_low < config.DELTA] = 0

    PG_violation = PG_violation_up + PG_violation_low

    # QG violation
    QG_violation_up = Pred_Qg - config.QG_Upbound
    QG_violation_up[QG_violation_up < config.DELTA] = 0

    QG_violation_low = config.QG_Lowbound - Pred_Qg
    QG_violation_low[QG_violation_low < config.DELTA] = 0

    QG_violation = QG_violation_up + QG_violation_low

    PG_violation_pp = -PG_violation_low + PG_violation_up
    QG_violation_pp = -QG_violation_low + QG_violation_up

    # ***** indicator function for PQ violation
    PG_violation_up[PG_violation_up < 0] = 0
    PG_violation_up[PG_violation_up > 0] = 1
    PG_violation_low[PG_violation_low < 0] = 0
    PG_violation_low[PG_violation_low > 0] = 1
    QG_violation_up[QG_violation_up < 0] = 0
    QG_violation_up[QG_violation_up > 0] = 1
    QG_violation_low[QG_violation_low < 0] = 0
    QG_violation_low[QG_violation_low > 0] = 1
    PG_violation_indicator = PG_violation_up - PG_violation_low
    QG_violation_indicator = QG_violation_up - QG_violation_low
    PQ_violation_indicator = np.concatenate((PG_violation_indicator, QG_violation_indicator), axis=1)

    return PG_violation, QG_violation, PQ_violation_indicator

def get_branch_violation(V):
    branch_temp = config.branch
    test_num = np.shape(V)[0]
    branch_violation = []
    branch_violation_count = 0
    branch_violation_ratio = 0
    for i in range(test_num):
        # current = admitance * voltage
        branch_temp[:, 2] = config.Rtest[i].T
        branch_temp[:, 3] = config.Xtest[i].T
        br = find(branch_temp[:, BR_STATUS]).astype(int)  ## in-service branches

        Ybus, Yf, Yt = makeYbus(config.baseMVA, config.bus, branch_temp)

        volt = V[i]

        Vf = volt[branch_temp[:, 0].astype(int)]
        If = Yf.dot(volt).conj()
        Ff = np.multiply(If, Vf)*config.baseMVA

        Vt = volt[branch_temp[:, 1].astype(int)]
        It = Yt.dot(volt).conj()
        Ft = np.multiply(It, Vt)*config.baseMVA
        ctol = 5e-06
        Branch_index_bound = np.where(branch_temp[:, 5] != 0)[0]
        RX_index = np.where(config.DN_test[i].T != 0)

        Branch_index = np.intersect1d(RX_index, Branch_index_bound)
        Branch_bound = branch_temp[Branch_index, 5] + ctol
        Ff = Ff.T
        Ff_violation = np.abs(Ff[Branch_index]) - Branch_bound
        Ff_violation[Ff_violation < 0] = 0

        Ft = Ft.T
        Ft_violation = np.abs(Ft[Branch_index]) - Branch_bound
        Ft_violation[Ft_violation < 0] = 0
        Ff_penalty = np.abs(Ff_violation)
        Ft_penalty = np.abs(Ft_violation)
        Branch_penalty = Ff_penalty + Ft_penalty
        branch_violation.append(Branch_penalty)

        if (np.sum(Branch_penalty) > 1e-4):
            branch_violation_count = branch_violation_count + np.count_nonzero(Branch_penalty)
            branch_violation_ratio = np.count_nonzero(Branch_penalty) / np.size(Branch_bound)

    branch_violation_ratio = branch_violation_ratio / (test_num) * 100

    return branch_violation_ratio

def get_V_violation(Pred_VM, Pred_Va):
    # PG violation
    Pred_VM = Pred_VM.clone().numpy()
    VM_Up_violation = Pred_VM - config.VmUb
    VM_Up_violation[VM_Up_violation < config.DELTA] = 0
    VM_Up_violation_bus = np.sum(VM_Up_violation, axis=1)

    VM_Low_violation = config.VmLb - Pred_VM
    VM_Low_violation[VM_Low_violation < config.DELTA] = 0
    VM_Low_violation_bus = np.sum(VM_Low_violation, axis=1)

    VM_violation_bus = VM_Up_violation_bus + VM_Low_violation_bus
    VM_violation = VM_Low_violation + VM_Up_violation
    VM_violation_num = np.count_nonzero(VM_violation)
    VM_violation_ratio = (VM_violation_num / (Pred_VM.shape[0]*Pred_VM.shape[1])) * 100

    Pred_VA = Pred_Va*180/np.pi

    VA_violation_ratio = 0
    for i in range(Pred_VA.shape[0]):
        RX_index = np.where(config.DN_test[i].T != 0)
        Pred_branch_angle = Pred_VA[i, config.branch[:, 0].astype(int)] - Pred_VA[i, config.branch[:, 1].astype(int)]
        VA_Up_violation = Pred_branch_angle[RX_index] - config.VA_Upbound[RX_index]
        VA_Up_violation[VA_Up_violation < config.DELTA] = 0
        VA_Up_violation_bus = np.sum(VA_Up_violation)
        VA_Low_violation = config.VA_Lowbound[RX_index] - Pred_branch_angle[RX_index]
        VA_Low_violation[VA_Low_violation < config.DELTA] = 0
        VA_Low_violation_bus = np.sum(VA_Low_violation)
        VA_violation_bus = VA_Up_violation_bus + VA_Low_violation_bus
        VA_violation = VA_Low_violation + VA_Up_violation
        VA_violation_num = np.count_nonzero(VA_violation)
        VA_violation_ratio = VA_violation_ratio + ((VA_violation_num / (Pred_branch_angle[RX_index].shape[0])) * 100)
    VA_violation_ratio = VA_violation_ratio / Pred_VA.shape[0]

    return VM_violation_ratio, VA_violation_ratio, VM_violation_bus, VA_violation_bus
