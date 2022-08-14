import MyPackage.config as config
from pypower.makeYbus import makeYbus
from pypower.loadcase import loadcase
from pypower.ext2int import ext2int1
import scipy.io
import numpy as np
import torch

def case_generation():

    mpc = scipy.io.loadmat(config.data_path + '/pglib_opf_case' + str(config.Nbus) + '_ieee' + '_para.mat')
    config.bus = mpc['bus']
    config.gen = mpc['gen']
    config.branch = mpc['branch']
    config.baseMVA = mpc['baseMVA'][0, 0]
    _, config.bus, config.gen, config.branch = ext2int1(config.bus, config.gen, config.branch)
    config.gencost = mpc['gencost']

    # *******   indices for slack bus, Pg and Qg bus  *********#
    config.bus_slack = np.where(config.bus[:, 1] == 3)
    config.bus_slack = np.squeeze(config.bus_slack)
    config.idxPg = np.squeeze(np.where(config.gen[:, 8] > 0), axis=0)
    config.idxQg = np.squeeze(np.where(config.gen[:, 3] > 0), axis=0)
    config.bus_Pg = config.gen[config.idxPg, 0].astype(int)
    config.bus_Qg = config.gen[config.idxQg, 0].astype(int)
    config.bus_PQg = np.concatenate((config.bus_Pg, config.bus_Qg + config.Nbus), axis=0)
    config.bus_PQg = torch.from_numpy(config.bus_PQg)

    # ************** Lower and upper bounds for Pg, Qg, Vm and Va ***********************#
    config.PG_Upbound = config.gen[config.idxPg, 8] / config.baseMVA
    config.PG_Lowbound = config.gen[config.idxPg, 9] / config.baseMVA
    config.QG_Upbound = config.gen[config.idxQg, 3] / config.baseMVA
    config.QG_Lowbound = config.gen[config.idxQg, 4] / config.baseMVA
    config.VA_Upbound = config.branch[:, 12]
    config.VA_Lowbound = config.branch[:, 11]

    # branch angle incidence matrix #
    config.BRANFT = torch.from_numpy(config.branch[:, 0:2]).long()
