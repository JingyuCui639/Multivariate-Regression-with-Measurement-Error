
import numpy as np

import simu_fun_collection3 as simuFun

import pandas as pd

####################################################################################
#Generalization of Simu 2: Non-normal distribution for ME, fix Cov of X as full type
#and vary the type of cov ME as "diagnal" and "full and not proportional"
#####################################################################################
#N_simu=10000
#N_est_simu=100000
case="X*Y"
dim_Y=4
dim_X=7
N_Obs=500
Sigma_Ex_type_list=["full and not proportional", "diagonal"]
error_level_list=[0.2,0.5,0.8]
U_distribution="Laplace"
X_distribution="chi-square"
diag_Ex_distribution="Gamma"

results_array=np.zeros((6,12))
i=0 
for Sigma_Ex_type in Sigma_Ex_type_list:
    for error_level in error_level_list:
        print(f"Sigma_Ex_type:{Sigma_Ex_type}")
        print(f"error_level:{error_level}")
        results_array[i,]=simuFun.main_simu_func(case, dim_Y, dim_X, cov_type="full", N_Obs=N_Obs, error_level=error_level,
                                        change="Sigma_Ex",Sigma_Ex_type=Sigma_Ex_type,
                                        X_distribution=X_distribution, diag_Ex_distribution=diag_Ex_distribution,
                                        U_distribution=U_distribution)
        i+=1

col_name_list=["avgBias_diff_B_LSE","Bhat_LSE_avgSEE","Bhat_LSE_avgSEM","Bhat_LSE_avgCR",	
            "avgBias_diff_B_naive","Bhat_naive_avgSEE","Bhat_naive_avgSEM","Bhat_naive_avgCR",	
            "avgBias_diff_B_correct","Bhat_correct_avgSEE","Bhat_correct_avgSEM","Bhat_correct_avgCR"]
               

results_array_df=pd.DataFrame(results_array, columns=col_name_list)  
results_array_df.to_csv("Simulation2_"+X_distribution+"_X_"+diag_Ex_distribution+"_Ex_"+U_distribution+"_U"+str(dim_Y)+str(dim_X)+"_N"+str(N_Obs)+"Simu100000.csv")
