
import numpy as np

import simu_fun_collection3 as simuFun

import pandas as pd

##########################
#simulation 3: IV estimator
##########################
results_array=np.zeros((3,12))

case="X*YV"
dim_Y=4
dim_X=7

cov_type_list=["full and not proportional", "sparse", "diagonal"]
N_Obs=1000 #or change to 500?

i=0
for cov_type in cov_type_list:
    print(f"Sigma_Ex type: {cov_type}")
    results_array[i,]=simuFun.main_simu_func(case, dim_Y, dim_X, "full", N_Obs, error_level=None, change="Sigma_Ex", Sigma_Ex_type=cov_type)
    i+=1

col_name_list=["avgBias_diff_B_LSE","Bhat_LSE_avgSEE","Bhat_LSE_avgSEM","Bhat_LSE_avgCR",	
            "avgBias_diff_B_naive","Bhat_naive_avgSEE","Bhat_naive_avgSEM","Bhat_naive_avgCR",	
            "avgBias_diff_B_inst","Bhat_inst_avgSEE","Bhat_inst_avgSEM","Bhat_inst_avgCR"]
                

results_array_df=pd.DataFrame(results_array, columns=col_name_list)  
results_array_df.to_csv("Simulation3_IV_estimate_vary_Sigma_Ex_"+str(dim_Y)+str(dim_X)+"_N"+str(N_Obs)+".csv")