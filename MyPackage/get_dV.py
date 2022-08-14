import MyPackage.config as config
import numpy as np
from pypower.makeYbus import makeYbus

# # # calculate dV using predicted V
# # lsidxtest is the test sample that has violation
def get_dV(Pred_V, PG_violation_gen, QG_violation_gen, PG_violation, QG_violation, bus_Pg, bus_Qg, PQ_violation_num):
     dV = np.zeros((PQ_violation_num, Pred_V.shape[1]*2))
     j = 0
     for i in range(Pred_V.shape[0]):
#         # determin whether there is violation
         if (PG_violation_gen[i] + QG_violation_gen[i]) >0:
#             # dS_dV
            V = Pred_V[i].copy()
            branch_temp = config.branch
            branch_temp[:, 2] = config.Rtest[i].T
            branch_temp[:, 3] = config.Xtest[i].T
            Ybus, Yf, Yt = makeYbus(config.baseMVA, config.bus, branch_temp)

            Ibus = Ybus.dot(V).conj()
            diagV = np.diag(V)
            diagIbus = np.diag(Ibus)
            diagVnorm = np.diag(V/np.abs(V))

            dSbus_dVm = np.dot(diagV, Ybus.dot(diagVnorm).conj()) + np.dot(diagIbus.conj(), diagVnorm)
            dSbus_dVa = 1j*np.dot(diagV, (diagIbus - Ybus.dot(diagV)).conj())

            dSbus_dV = np.concatenate((dSbus_dVa, dSbus_dVm), axis=1)
            dPbus_dV = np.real(dSbus_dV)
            dQbus_dV = np.imag(dSbus_dV)
#
            dPQGbus_dV = np.concatenate((dPbus_dV[bus_Pg, :], dQbus_dV[bus_Qg, :]), axis=0) #need bus number pf Pg Qg
            dPQg = np.concatenate((PG_violation[i,:], QG_violation[i, :]), axis=0)

            dV[j] = np.dot(np.linalg.pinv(dPQGbus_dV), dPQg * config.k_dV)
            j+=1
     return dV
