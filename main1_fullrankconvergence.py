#import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA #this package is used when calculating the norm
import simu_fun_collection1 as fun_old
import simu_fun_collection as fun

N_list=[100,500, 1000,5000, 10000,50000, 100000, 500000, 1000000]
N_simu=100
M=len(N_list)
B_ls_norm_diff=np.zeros((N_simu, M))
B_xerr_norm_diff=np.zeros((N_simu, M))
B_yerr_norm_diff=np.zeros((N_simu, M))
B_xyerr_norm_diff=np.zeros((N_simu, M))
for simu_id in range(N_simu):
    i=0
    for N in N_list:
        X, Y, B, X_err, Y_err, B_star, cov_x, COV_Y, cov_x_err, cov_y_err, COV_zz=fun_old.data_generator(N_VARIABLES_X=64, N_VARIABLES_Y=36, N_OBS=N,
        MODEL={"X": "exp","Y": "exp"}, MODEL_INDEX={"X_IND":1, "Y_IND": 0.8}, RANK_COEFF=36, N2S_RATIO_Z=0.5, N2S_RATIO_X=0.3, N2S_RATIO_Y=0.3)
        B_ls=fun_old.LS_fit(X, Y)
        B_ls_norm_diff[simu_id, i]=LA.norm(B_ls-B)
        B_xerr=fun_old.LS_fit(X_err, Y)
        B_xerr_norm_diff[simu_id, i]=LA.norm(B_xerr-B_star)
        B_yerr=fun_old. LS_fit(X, Y_err)
        B_yerr_norm_diff[simu_id, i]=LA.norm(B_yerr-B)
        B_xyerr=fun_old.LS_fit(X_err, Y_err)
        B_xyerr_norm_diff[simu_id, i]=LA.norm(B_xyerr-B_star)
        i+=1

fullrankB_ls=B_ls_norm_diff.mean(axis=0)
fullrankBxerr=B_xerr_norm_diff.mean(axis=0)
fullrankByerr=B_yerr_norm_diff.mean(axis=0)
fullrankBxyerr=B_xyerr_norm_diff.mean(axis=0)

fig, axs = plt.subplots()
x_axis=np.log10(N_list)
axs.plot(x_axis,fullrankB_ls, color='orange',linestyle='-', label='||B_xy-B||_F')
axs.plot(x_axis,fullrankBxerr, color='blue',linestyle='--' ,label='||B_x*-B*||_F')
axs.plot(x_axis,fullrankByerr, color='green', linestyle='-.',label='||B_y*-B||_F')
axs.plot(x_axis,fullrankBxyerr, color='red',linestyle=':', label='||B_x*y*-B*||_F')
axs.axhline(y=0, color='black', linestyle='-')
#axs.set_xticklabels([2,3,4,5,6,6.72])
axs.set_xlabel('Sample size n: (log10(n))')
axs.legend(loc="upper right")
plt.show()