import numpy as np
from numpy.lib.function_base import append
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA #this package is used when calculating the norm
import simu_fun_collection_new as fun

#Step 1 generate a fixed B
N_VARIABLES_X=36
N_VARIABLES_Y=16
MODEL={"X": "exp","Y": "exp"}
MODEL_INDEX={"X_IND":1, "Y_IND": 0.8}
RANK_COEFF=10
N2S_RATIO_Z=0.1
N2S_RATIO_X=0.1
N2S_RATIO_Y=0.1

N_N_SIMU=50
n=5000
cov_x=fun.cov_generator(N_VARIABLES=N_VARIABLES_X, MODEL=MODEL["X"], space_index=MODEL_INDEX["X_IND"])
cov_y=fun.cov_generator(N_VARIABLES=N_VARIABLES_Y, MODEL=MODEL["Y"], space_index=MODEL_INDEX["Y_IND"])
B=fun.coeff_generator(COV_X=cov_x, COV_Y=cov_y,RANK=RANK_COEFF)

#loop among sample sizes from 100 to 1000000
N_SIMU_list=[10, 100, 1000, 10000,  100000]

accurate_norm_diffB=[]
accurate_norm_diffCOV=[]
Xerror_norm_diffB=[]
Xerror_norm_diffCOV=[]
Yerror_norm_diffB=[]
Yerror_norm_diffCOV=[]
XYerror_norm_diffB=[]
XYerror_norm_diffCOV=[]

for n_simu in N_SIMU_list:

    length1=N_VARIABLES_X*N_VARIABLES_Y
    length2=(N_VARIABLES_X*N_VARIABLES_Y)**2

    accurate_B_diff=np.zeros((length1,N_N_SIMU))
    accurate_cov_diff=np.zeros((length2, N_N_SIMU))
    Xerror_B_diff=np.zeros((length1,N_N_SIMU))
    Xerror_cov_diff=np.zeros((length2, N_N_SIMU))
    Yerror_B_diff=np.zeros((length1,N_N_SIMU))
    Yerror_cov_diff=np.zeros((length2, N_N_SIMU))
    XYerror_B_diff=np.zeros((length1,N_N_SIMU))
    XYerror_cov_diff=np.zeros((length2, N_N_SIMU))

    for i in range(N_N_SIMU):

        mean_diff_vec_B, diff_cov=fun.low_rank_simu(N_VARIABLES_X,N_VARIABLES_Y, 
            B_fixed=B, N_OBS=n,N_simu=n_simu, DATATYPE="accurate",MODEL={"X": "exp","Y": "exp"}, 
            MODEL_INDEX={"X_IND":1, "Y_IND": 0.8}, RANK_COEFF=RANK_COEFF, RANK_EST=RANK_COEFF, N2S_RATIO_Z=N2S_RATIO_Z, N2S_RATIO_X=N2S_RATIO_X, N2S_RATIO_Y=N2S_RATIO_Y)

        accurate_B_diff[:,i]=mean_diff_vec_B
        accurate_cov_diff[:,i]=diff_cov

        mean_diff_vec_B, diff_cov=fun.low_rank_simu(N_VARIABLES_X,N_VARIABLES_Y, 
            B_fixed=B, N_OBS=n,N_simu=n_simu, DATATYPE="error-prone X",MODEL={"X": "exp","Y": "exp"}, 
            MODEL_INDEX={"X_IND":1, "Y_IND": 0.8}, RANK_COEFF=RANK_COEFF, RANK_EST=RANK_COEFF,  N2S_RATIO_Z=N2S_RATIO_Z, N2S_RATIO_X=N2S_RATIO_X, N2S_RATIO_Y=N2S_RATIO_Y)

        Xerror_B_diff[:,i]=mean_diff_vec_B
        Xerror_cov_diff[:,i]=diff_cov

        mean_diff_vec_B, diff_cov=fun.low_rank_simu(N_VARIABLES_X,N_VARIABLES_Y, 
            B_fixed=B, N_OBS=n,N_simu=n_simu, DATATYPE="error-prone Y",MODEL={"X": "exp","Y": "exp"}, 
            MODEL_INDEX={"X_IND":1, "Y_IND": 0.8}, RANK_COEFF=RANK_COEFF, RANK_EST=RANK_COEFF, N2S_RATIO_Z=N2S_RATIO_Z, N2S_RATIO_X=N2S_RATIO_X, N2S_RATIO_Y=N2S_RATIO_Y)

        Yerror_B_diff[:,i]=mean_diff_vec_B
        Yerror_cov_diff[:,i]=diff_cov

        mean_diff_vec_B, diff_cov=fun.low_rank_simu(N_VARIABLES_X,N_VARIABLES_Y, 
            B_fixed=B, N_OBS=n,N_simu=n_simu, DATATYPE="error-prone Y",MODEL={"X": "exp","Y": "exp"}, 
            MODEL_INDEX={"X_IND":1, "Y_IND": 0.8}, RANK_COEFF=RANK_COEFF, RANK_EST=RANK_COEFF,  N2S_RATIO_Z=N2S_RATIO_Z, N2S_RATIO_X=N2S_RATIO_X, N2S_RATIO_Y=N2S_RATIO_Y)

        XYerror_B_diff[:,i]=mean_diff_vec_B
        XYerror_cov_diff[:,i]=diff_cov

    accurate_mean_diffB_oversimu=np.mean(accurate_B_diff, axis=1)
    accurate_mean_diffCOV_oversimu=np.mean(accurate_cov_diff, axis=1)
    Xerror_mean_diffB_oversimu=np.mean(Xerror_B_diff, axis=1)
    Xerror_mean_diffCOV_oversimu=np.mean(Xerror_cov_diff, axis=1)
    Yerror_mean_diffB_oversimu=np.mean(Yerror_B_diff, axis=1)
    Yerror_mean_diffCOV_oversimu=np.mean(Yerror_cov_diff, axis=1)
    XYerror_mean_diffB_oversimu=np.mean(XYerror_B_diff, axis=1)
    XYerror_mean_diffCOV_oversimu=np.mean(XYerror_cov_diff, axis=1)

    accurate_norm_diffB.append(np.sqrt(LA.norm(accurate_mean_diffB_oversimu)**2/length1))
    accurate_norm_diffCOV.append(np.sqrt(LA.norm(accurate_mean_diffCOV_oversimu)**2/length2))
    Xerror_norm_diffB.append(np.sqrt(LA.norm(Xerror_mean_diffB_oversimu)**2/length1))
    Xerror_norm_diffCOV.append(np.sqrt(LA.norm(Xerror_mean_diffCOV_oversimu)**2/length2))
    Yerror_norm_diffB.append(np.sqrt(LA.norm(Yerror_mean_diffB_oversimu)**2/length1))
    Yerror_norm_diffCOV.append(np.sqrt(LA.norm(Yerror_mean_diffCOV_oversimu)**2/length2))
    XYerror_norm_diffB.append(np.sqrt(LA.norm(XYerror_mean_diffB_oversimu)**2/length1))
    XYerror_norm_diffCOV.append(np.sqrt(LA.norm(XYerror_mean_diffCOV_oversimu)**2/length2))

# Given the sample size =5000
# As the simulation number is increasing, the difference between the estimates and the corresponding converged results

fig1, axs1 = plt.subplots()
x_axis=np.log10(N_SIMU_list)
axs1.plot(x_axis,accurate_norm_diffB, color='orange',linestyle='-', label='Accurate $X, Y$')
axs1.plot(x_axis,Xerror_norm_diffB, color='blue',linestyle='--' ,label='$X^*, Y$')
axs1.plot(x_axis,Yerror_norm_diffB, color='green', linestyle='-.',label='$X,Y^*$')
axs1.plot(x_axis,XYerror_norm_diffB, color='red',linestyle=':', label='$X^*,Y^*$')
axs1.axhline(y=0, color='grey', linestyle='-')
axs1.set_xlabel('Simulation number: log10(#)')
axs1.legend(loc="upper right")
axs1.title("Given sample size N=5000, the B convergence as the simulation number increases.")
plt.show()

fig2, axs2 = plt.subplots()
x_axis=np.log10(N_SIMU_list)
axs2.plot(x_axis,accurate_norm_diffCOV, color='orange',linestyle='-', label='Accurate X, Y')
axs2.plot(x_axis,Xerror_norm_diffCOV, color='blue',linestyle='--' ,label='$X^*, Y$')
axs2.plot(x_axis,Yerror_norm_diffCOV, color='green', linestyle='-.',label='$X,Y^*$')
axs2.plot(x_axis,XYerror_norm_diffCOV, color='red',linestyle=':', label='$X^*,Y^*$')
axs2.axhline(y=0, color='grey', linestyle='-')
axs2.set_xlabel('Simulation number: log10(#)')
axs2.legend(loc="upper right")
axs2.title("Given sample size N=5000, the COV convergence as the simulation number increases.")
plt.show()
