#############################
##Simulation for Chapter 8####
#############################

import numpy as np

import simu_fun_collection3 as simuFun

import pandas as pd

N_simu=10000

#N_est_simu=100000

#####
# !!!!!! If want the curren draft results, please set the below to false
####
chapter_8=True # this changes the true B and make Sigam_X and Sigma_E's diagonals not the same, so K's diagonals are not the same

case="X*Y"
dim_Y=4
dim_X=7
N_Obs=10000 #500, 2000, 10000
cov_type="identity"# # indicate the cov type of X: "full" or "identity"
Sigma_Ex_type="diagonal"# "full and not proportional" or "diagonal"
error_level_list=[0.2,0.5,0.8]
distribution_list=[["Normal", "Normal", "Normal"], ["Uniform", "Gamma", "chi-square"]]
correct_method = "estimated K"
optimization_algrithm="diagonalK" # "diagonalK" or "TNC" or "prewhitten"

# X_distribution="Uniform"
# diag_Ex_distribution="Gamma"
# U_distribution="chi-square"

results_array=np.zeros((6,12))
i=0 
for distribution_comb in distribution_list:
    
    X_distribution,diag_Ex_distribution,U_distribution=distribution_comb
    
    print(f"X_distribution:{X_distribution}")
    print(f"diag_Ex_distribution:{diag_Ex_distribution}")
    print(f"U_distribution:{U_distribution}")
    
    for error_level in error_level_list:
        
        print(f"error_level:{error_level}")
        
        results_array[i,]=simuFun.main_simu_func(case, dim_Y, dim_X, cov_type=cov_type, N_Obs=N_Obs, error_level=error_level,
                                        change="Sigma_Ex",Sigma_Ex_type=Sigma_Ex_type,
                                        X_distribution=X_distribution, diag_Ex_distribution=diag_Ex_distribution,
                                        U_distribution=U_distribution, correct_method="estimated K", N_Simu=N_simu, 
                                        opt_method=optimization_algrithm, save_estimates=False, chap_8=chapter_8)
        i+=1

col_name_list=["RD (naive)","avgBias (naive)","avgSEE (naive)","avgMSE (naive)",
               "RD (corrected)","avgBias (corrected)","avgSEE (corrected)","avgMSE (corrected)",
               "RD (LSE)","avgBias (LSE)","avgSEE (LSE)","avgMSE (LSE)"]
               

results_array_df=pd.DataFrame(results_array, columns=col_name_list)  
#below line is the save file name for the current draft version
#results_array_df.to_csv("Simulation_Chapter8_p"+str(dim_Y)+"_q"+str(dim_X)+"_N"+str(N_Obs)+"_Simu"+str(N_simu)+"_"+optimization_algrithm+"_Diags2CovX_DiagCovEx_diverging.csv")

#the line below is the revised version: change B and make K not identity
results_array_df.to_csv("Simulation_Chapter8_p"+str(dim_Y)+"_q"+str(dim_X)+"_N"+str(N_Obs)+"_Simu"+str(N_simu)+"_"+optimization_algrithm+"_ChangeBK.csv")
