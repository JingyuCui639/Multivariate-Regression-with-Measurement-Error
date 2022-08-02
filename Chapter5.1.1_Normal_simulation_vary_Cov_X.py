
import numpy as np

import simu_fun_collection3 as simuFun

import pandas as pd


#################################################################
# simulation 1: vary cov of X and fix the cov of measurement eror
#################################################################
case="X*Y"
dim_Y=3
dim_X=3
N_Obs=500
N_simu=10000
N_est_simu=100000
cov_type_list=["full", "sparse", "identity"]
N_Obs_list=[100, 500, 1000]
error_level_list=[0.2, 0.5, 0.8]
results_array=np.zeros((27,21))

i=0
for cov_type in cov_type_list:
    
    for N_Obs in N_Obs_list:
        
        for error_level in error_level_list:
            
            print(f"cov_type: {cov_type}")
            print(f"N_Obs:{N_Obs}")
            print(f"error_level: {error_level}")
            
            results_array[i,]=simuFun.main_simu_func(case, dim_Y, dim_X, cov_type, N_Obs, error_level)
            
            i+=1
            
col_name_list=["avgBias_diff_B_trueX","Bhat_LSE_avgSEE","Bhat_LSE_avgSEM","Bhat_LSE_avgCR",#"Bhat_LSE_Frob","Bhat_LSE_infinity_norm",	
            "avgBias_diff_B_naive","Bhat_naive_avgSEE","Bhat_naive_avgSEM","Bhat_naive_avgCR",#"Bhat_naive_Frob","Bhat_naive_infinity_norm",	
            "avgBias_diff_B_correct","Bhat_correct_avgSEE","Bhat_correct_avgSEM","Bhat_correct_avgCR",#"Bhat_correct_Frob","Bhat_correct_infinity_norm",	
            #"avgBias_diff_B_correct_estCov",	"Bhat_correct2_avgSEE","Bhat_correct2_Frob","Bhat_correct2_infinity_norm",
            "Cov_LSE_avgBias","Cov_LSE_Frob","Cov_LSE_infinity_norm",	
            "Cov_naive_avgBias",	"Cov_naive_Frob",	"Cov_naive_infinity_norm",	
            "Cov_correct_avgBias","Cov_correct_Frob","Cov_correct_infinity_norm"]

results_array_df=pd.DataFrame(results_array, columns=col_name_list)  
results_array_df.to_csv("Simulation1_results_vary_SigmaX_"+str(dim_Y)+str(dim_X)+".csv")

