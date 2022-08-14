import MyPackage.config as config
import numpy as np
from pypower.makeYbus import makeYbus
import torch
def get_relative_error(real, predict):
    '''
    relative error
    '''
    if len(real) == len(predict):
        err = (np.sum(predict) - np.sum(real)) / (np.sum(real)) * 100
        return err
    else:
        return None

def get_load_mismatch_rate(real, predict):
    '''
    absolute relative error
    '''
    if len(real) == len(predict):
        err = abs((real - predict) / real) * 100
        return err
    else:
        return None

## other function
def get_clamp(Pred, Predmin, Predmax):
    # each row is a sample;Predmin and Predmax is the limit for each element of each row
    Pred_clip = Pred.clone()
    for i in range(Pred.shape[1]):
        Pred_clip[:, i] = Pred_clip[:, i].clamp(min=Predmin[i])
        Pred_clip[:, i] = Pred_clip[:, i].clamp(max=Predmax[i])
    return Pred_clip

def get_abs_error(real, predict):
    '''
    mean absolute error
    '''
    if len(real) == len(predict):
        err = torch.mean(torch.abs(real - predict))
        return err
    else:
        return None

def get_genload(V, Pdtest, Qdtest, bus_Pg, bus_Qg):

    S = np.zeros(V.shape, dtype='complex_')
    branch_PQ = config.branch

    for i in range(V.shape[0]):
        # current = admitance * voltage
        branch_PQ[:, 2] = config.Rtest[i].T
        branch_PQ[:, 3] = config.Xtest[i].T
        Ybus, Yf, Yt = makeYbus(config.baseMVA, config.bus, branch_PQ)
        I = Ybus.dot(V[i]).conj()
        # S = current * voltage
        S[i] = np.multiply(V[i], I)

    P = np.real(S)
    Q = np.imag(S)
    Pg = P[:, bus_Pg] + Pdtest[:, bus_Pg]
    Qg = Q[:, bus_Qg] + Qdtest[:, bus_Qg]

    Pd = -P * 1.0
    Qd = -Q * 1.0
    Pd[:, bus_Pg] = Pg - P[:, bus_Pg]
    Qd[:, bus_Qg] = Qg - Q[:, bus_Qg]
    return Pg, Qg, Pd, Qd

# cost
def get_Pgcost(Pg, idxPg, gencost):
    cost = np.zeros(Pg.shape[0])
    PgMVA = Pg * config.baseMVA
    for i in range(Pg.shape[0]):
        c1 = np.multiply(gencost[idxPg, 4], np.multiply(PgMVA[i, :], PgMVA[i, :]))   # quadratic term
        c2 = np.multiply(gencost[idxPg, 5], PgMVA[i, :])   # linear term
        c3 = gencost[idxPg, 6]  # constant term
        cost[i] = np.sum(c1 + c2 + c3)

    return cost