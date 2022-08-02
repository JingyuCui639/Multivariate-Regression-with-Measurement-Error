
import numpy as np

import simu_fun_collection3 as simuFun

import pandas as pd

#################################################################
#simulation 2: fix the cov of X and vary cov of measurement error
#################################################################
case="X*Y"
dim_Y=4
dim_X=7
N_Obs=500
N_simu=10000
N_est_simu=100000
cov_type_list=["full and proportional", "full and not proportional", "sparse", "diagonal"]

results_array=np.zeros((4,21))

i=0
for cov_type in cov_type_list:
    print(f"Sigma_Ex type: {cov_type}")
    results_array[i,]=simuFun.main_simu_func(case, dim_Y, dim_X, "full", N_Obs, error_level=0.5, change="Sigma_Ex", Sigma_Ex_type=cov_type)
    i+=1
    
            
col_name_list=["avgBias_diff_B_trueX","Bhat_LSE_avgSEE","Bhat_LSE_avgSEM","Bhat_LSE_avgCR",#"Bhat_LSE_Frob","Bhat_LSE_infinity_norm",	
            "avgBias_diff_B_naive","Bhat_naive_avgSEE","Bhat_naive_avgSEM","Bhat_naive_avgCR",#"Bhat_naive_Frob","Bhat_naive_infinity_norm",	
            "avgBias_diff_B_correct","Bhat_correct_avgSEE","Bhat_correct_avgSEM","Bhat_correct_avgCR",#"Bhat_correct_Frob","Bhat_correct_infinity_norm",	
            #"avgBias_diff_B_correct_estCov",	"Bhat_correct2_avgSEE","Bhat_correct2_Frob","Bhat_correct2_infinity_norm",
            "Cov_LSE_avgBias","Cov_LSE_Frob","Cov_LSE_infinity_norm",	
            "Cov_naive_avgBias",	"Cov_naive_Frob",	"Cov_naive_infinity_norm",	
            "Cov_correct_avgBias","Cov_correct_Frob","Cov_correct_infinity_norm"]

results_array_df=pd.DataFrame(results_array, columns=col_name_list)  
results_array_df.to_csv("Simulation2_results_vary_Sigma_Ex_"+str(dim_Y)+str(dim_X)+".csv")