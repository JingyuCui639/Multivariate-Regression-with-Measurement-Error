from cmath import sqrt
import sys
import numpy as np
from numpy.lib.function_base import append
import matplotlib.pyplot as plt
from numpy import linalg as LA
import simu_fun_collection3 as simuFun
import seaborn as sns
import scipy.stats as ss

def single_simulation_accurate(case,dim_Y, dim_X, N_Obs, COV_X, COV_U, COV_Ex, 
                            COV_Ey, COV_R, COV_V, B, A=0):
    """
        case: "XY","X*Y","XY*","X*Y*"
    """

    if case=="X*YV":

              Y_data, X_data, V_data=simuFun.data_generation(case=case,dim_Y=dim_Y, dim_X=dim_X, N_Obs=N_Obs,
                            COV_X=COV_X, COV_U=COV_U, COV_Ex=COV_Ex,COV_Ey=COV_Ey, COV_R=COV_R,
                            COV_V=COV_V, B=B,A=A)
              
              S_YV=Y_data@V_data.T/N_Obs
              S_XV=X_data@V_data.T/N_Obs
              B_hat=S_YV@LA.inv(S_XV)
              return B_hat
    else:

              #Step 2. data generation

              Y_data,X_data=simuFun.data_generation(case=case,dim_Y=dim_Y, dim_X=dim_X, N_Obs=N_Obs,
                                          COV_X=COV_X, COV_U=COV_U, COV_Ex=COV_Ex,COV_Ey=COV_Ey, COV_R=COV_R,
                                          COV_V=COV_V, B=B,A=A)
                            

              #Step 3. calculate the naive estimator
              #empirical Sigma_YX
              S_YX=Y_data@X_data.T/N_Obs
              #empirical Sigma_X
              S_XX=X_data@X_data.T/N_Obs
              #Estimating B
              B_hat_naive=S_YX@LA.inv(S_XX)

              #Step 4. Calculate the corrected estimator
              K=(COV_X+COV_Ex)@LA.inv(COV_X)
              B_hat_correct=B_hat_naive@K

              #Step 3. return the estimated B
              return B_hat_naive, B_hat_correct


if __name__=="__main__":
    
    #obtain the inputs
    case=sys.argv[1]
    dim_Y=int(sys.argv[2])
    dim_X=int(sys.argv[3])

    N_Obs=3000
    N_Simu=10000

    #Step 1. generate fixed parameters

    B=np.random.uniform(low=-5, high=5, size=(dim_Y, dim_X))

    A=np.random.uniform(low=-0.5, high=2, size=(dim_X,dim_X))

    Sigma_X=simuFun.cov_generator(dim=dim_X, MODEL="random")

    Sigma_V=simuFun.cov_generator(dim=dim_X, MODEL="random")

    Sigma_U=simuFun.cov_generator(dim=dim_Y, MODEL="identity", identity_sd=[1])

    if case=="XY":

        Sigma_Ex=0
        Sigma_Ey=0
        Sigma_R=0
    
    elif case=="X*Y":

        Sigma_Ex=simuFun.cov_generator(dim=dim_X, MODEL="identity", identity_sd=[0.8])
        Sigma_Ey=0
        Sigma_R=0

    elif case=="XY*":

        Sigma_Ex=0
        Sigma_Ey=simuFun.cov_generator(dim=dim_Y, MODEL="identity", identity_sd=[0.8])
        Sigma_R=0

    elif case=="X*Y*":

        Sigma_Ex=simuFun.cov_generator(dim=dim_X, MODEL="identity", identity_sd=[0.8])
        Sigma_Ey=simuFun.cov_generator(dim=dim_Y, MODEL="identity", identity_sd=[0.8])
        Sigma_R=0

    elif case=="X*YV":
        Sigma_Ex=simuFun.cov_generator(dim=dim_X, MODEL="identity", identity_sd=[0.8])
        Sigma_Ey=0
        Sigma_R=simuFun.cov_generator(dim=dim_X, MODEL="identity", identity_sd=[0.5])

    
    #Step 2. simulations
    if case=="XY" or case=="XY*":
        
        sum_B=0
        vec_Bhat_array=np.zeros((dim_Y*dim_X,N_Simu))

        for simu_id in range(N_Simu):

              B_hat, _=single_simulation_accurate(case=case,dim_Y=dim_Y, dim_X=dim_X, N_Obs=N_Obs, 
                                          COV_X=Sigma_X, COV_U=Sigma_U, COV_Ex=Sigma_Ex, COV_Ey=Sigma_Ey, 
                                          COV_R=Sigma_R, COV_V=Sigma_V, B=B)
              
              sum_B+=B_hat
              vec_Bhat_array[:,simu_id]=B_hat.T.reshape(-1)
        #After simulation, calculate summarized results

        #plt.imshow(B)

        #1. biases of averaged average B_hat
        mean_B_hat=sum_B/N_Simu
        diff_B=mean_B_hat-B

        #calculate the R2 between \bar{\hat{B}} and true B
        R2_B=1-LA.norm(diff_B)**2/LA.norm(B)**2
        print(f'The $R^2$ on B is: {R2_B}.')
    
        #2. empirical cov
        vec_mean_B=mean_B_hat.T.reshape(-1)
        centered_vec_B=vec_Bhat_array-np.array([vec_mean_B,]*N_Simu).T
        empirical_cov=centered_vec_B@centered_vec_B.T/(N_Simu-1)
        if case=="XY":
                      theoretical_cov=np.kron(LA.inv(Sigma_X),Sigma_U)/N_Obs
        elif case=="XY*":
                      theoretical_cov=np.kron(LA.inv(Sigma_X),(Sigma_U+Sigma_Ey))/N_Obs
        diff_cov=empirical_cov-theoretical_cov
        #calculate the R2 on cov
        R2_cov=1-LA.norm(empirical_cov-theoretical_cov)**2/LA.norm(theoretical_cov)**2
        print(f'The $R^2$ on covariance: {R2_cov}.')

        #4. calculate the CR% of 95% CI for each element of B
        vec_B=B.T.reshape(-1)
        vec_B_matrix=np.array([vec_B,]*N_Simu).T
        Var_B=np.diag(empirical_cov)
        sd_B_matrix=np.array([np.sqrt(Var_B),]*N_Simu).T
        lower_bound=vec_Bhat_array+ss.norm.ppf(0.025)*sd_B_matrix
        upper_bound=vec_Bhat_array+ss.norm.ppf(0.975)*sd_B_matrix
        InCI_Logical=np.greater(vec_B_matrix,lower_bound) & np.less(vec_B_matrix, upper_bound)
        InCI_numeric=InCI_Logical*1
        InCI=np.mean(InCI_numeric,axis=1)*100
        CR=InCI.reshape(dim_X, dim_Y).T
        #plt.imshow(CR)

        #3. calculate the 2ed moment matrix 
        empirical_2ed_m=vec_Bhat_array@vec_Bhat_array.T/N_Simu
        converg_valu_2=np.outer(vec_B,vec_B)
        theoretical_2ed_m=theoretical_cov+converg_valu_2
        diff_2ed_m=empirical_2ed_m-theoretical_2ed_m
        #calculate the R2 of 2ed moment
        R2_2ed_m=1-LA.norm(diff_2ed_m)**2/LA.norm(theoretical_2ed_m)**2
        print(f'The $R^2$ on 2ed moment: {R2_2ed_m}.')

        fig, axs = plt.subplots(2, 2, figsize=(18,5), dpi=100)
         
        sns.heatmap(diff_B, ax=axs[0, 0], cmap='PiYG') 
        axs[0, 0].set_title('Delta B')
        sns.heatmap(diff_cov, ax=axs[0, 1], cmap='PRGn')
        axs[0, 1].set_title('Difference covariance')
        sns.heatmap(CR, ax=axs[1,0], cmap='Spectral')
        axs[1,0].set_title('95% CR')
        sns.heatmap(diff_2ed_m, ax=axs[1, 1], cmap='BrBG')
        axs[1, 1].set_title('Difference 2ed moment')

        plt.show()


    elif case== "X*Y" or case=="X*Y*":
        
        sum_B_naive=0
        vec_Bhat_array_naive=np.zeros((dim_Y*dim_X,N_Simu))
        sum_B_correct=0
        vec_Bhat_array_correct=np.zeros((dim_Y*dim_X,N_Simu))
        for simu_id in range(N_Simu):

              B_hat_naive, B_hat_correct=single_simulation_accurate(case=case,dim_Y=dim_Y, dim_X=dim_X, N_Obs=N_Obs, 
                                          COV_X=Sigma_X, COV_U=Sigma_U, COV_Ex=Sigma_Ex, COV_Ey=Sigma_Ey,
                                          COV_R=Sigma_R, COV_V=Sigma_V, B=B)
              
              sum_B_naive+=B_hat_naive
              vec_Bhat_array_naive[:,simu_id]=B_hat_naive.T.reshape(-1)
              sum_B_correct+=B_hat_correct
              vec_Bhat_array_correct[:,simu_id]=B_hat_correct.T.reshape(-1)
        #After simulation, calculate summarized results

        #plt.imshow(B)

        #1. biases of averaged average B_hat
        #naive
        mean_B_hat_naive=sum_B_naive/N_Simu
        diff_B_naive=mean_B_hat_naive-B
        #correct
        mean_B_hat_correct=sum_B_correct/N_Simu
        diff_B_correct=mean_B_hat_correct-B

        #calculate the R2 between \bar{\hat{B}} and true B
        R2_B_naive=1-LA.norm(mean_B_hat_naive-B)**2/LA.norm(B)**2
        print(f'The R2 for naive B is: {R2_B_naive}.')
        R2_B_correct=1-LA.norm(mean_B_hat_correct-B)**2/LA.norm(B)**2
        print(f'The R2 for the corrected B is: {R2_B_correct}.')
    
        #2. empirical cov of B_naive and B_correct
        # naive B
        vec_mean_B_naive=mean_B_hat_naive.T.reshape(-1)
        centered_vec_B_naive=vec_Bhat_array_naive-np.array([vec_mean_B_naive,]*N_Simu).T
        empirical_cov_naive=centered_vec_B_naive@centered_vec_B_naive.T/(N_Simu-1)
        # theoretical cov of naive estimator
        

        #corrected B
        vec_mean_B_correct=mean_B_hat_correct.T.reshape(-1)
        centered_vec_B_correct=vec_Bhat_array_correct-np.array([vec_mean_B_correct,]*N_Simu).T
        empirical_cov_correct=centered_vec_B_correct@centered_vec_B_correct.T/(N_Simu-1)

#         theoretical_cov=np.kron(LA.inv(Sigma_X),Sigma_U)/N_Obs
#         diff_cov=empirical_cov-theoretical_cov
#         #calculate the R2 on cov
#         R2_cov=1-LA.norm(empirical_cov-theoretical_cov)/LA.norm(theoretical_cov)
#         print(f'The $R^2$ on covariance: {R2_cov}.')

        #4. calculate the CR% of 95% CI for each element of B
        vec_B=B.T.reshape(-1)
        vec_B_matrix=np.array([vec_B,]*N_Simu).T
        #naive case
        Var_B_naive=np.diag(empirical_cov_naive)
        sd_B_matrix_naive=np.array([np.sqrt(Var_B_naive),]*N_Simu).T
        lower_bound_naive=vec_Bhat_array_naive+ss.norm.ppf(0.025)*sd_B_matrix_naive
        upper_bound_naive=vec_Bhat_array_naive+ss.norm.ppf(0.975)*sd_B_matrix_naive
        InCI_Logical_naive=np.greater(vec_B_matrix,lower_bound_naive) & np.less(vec_B_matrix, upper_bound_naive)
        InCI_numeric_naive=InCI_Logical_naive*1
        InCI_naive=np.mean(InCI_numeric_naive,axis=1)*100
        CR_naive=InCI_naive.reshape(dim_X, dim_Y).T
        #correct
        Var_B_correct=np.diag(empirical_cov_correct)
        sd_B_matrix_correct=np.array([np.sqrt(Var_B_correct),]*N_Simu).T
        lower_bound_correct=vec_Bhat_array_correct+ss.norm.ppf(0.025)*sd_B_matrix_correct
        upper_bound_correct=vec_Bhat_array_correct+ss.norm.ppf(0.975)*sd_B_matrix_correct
        InCI_Logical_correct=np.greater(vec_B_matrix, lower_bound_correct) & np.less(vec_B_matrix,upper_bound_correct)
        InCI_numeric_correct=InCI_Logical_correct*1
        InCI_correct=np.mean(InCI_numeric_correct, axis=1)*100
        CR_correct=InCI_correct.reshape(dim_X, dim_Y).T

        #3. calculate the 2ed moment matrix 
        #naive
        empirical_2ed_m_naive=vec_Bhat_array_naive@vec_Bhat_array_naive.T/N_Simu
        if case=="X*Y":
                      theoretical_2ed_m_naive=np.kron(LA.inv(Sigma_X+Sigma_Ex),(B@Sigma_X@B.T+Sigma_U))/N_Obs
        elif case=="X*Y*":
                      theoretical_2ed_m_naive=np.kron(LA.inv(Sigma_X+Sigma_Ex),(B@Sigma_X@B.T+Sigma_U+Sigma_Ey))/N_Obs
        diff_2ed_m_naive=empirical_2ed_m_naive-theoretical_2ed_m_naive
        #calculate the R2 of 2ed moment
        R2_2ed_m_naive=1-LA.norm(diff_2ed_m_naive)**2/LA.norm(theoretical_2ed_m_naive)**2
        print(f'The R2 of 2ed moment for naive estimator: {R2_2ed_m_naive}.')
        #correct
        empirical_2ed_m_correct=vec_Bhat_array_correct@vec_Bhat_array_correct.T/N_Simu
        if case=="X*Y":
                      theoretical_2ed_m_correct=np.kron(LA.inv(Sigma_X)@(Sigma_X+Sigma_Ex)@LA.inv(Sigma_X),(B@Sigma_X@B.T+Sigma_U))/N_Obs
        elif case=="X*Y*":
                      theoretical_2ed_m_correct=np.kron(LA.inv(Sigma_X)@(Sigma_X+Sigma_Ex)@LA.inv(Sigma_X),(B@Sigma_X@B.T+Sigma_U+Sigma_Ey))/N_Obs
        diff_2ed_m_correct=empirical_2ed_m_correct-theoretical_2ed_m_correct
        #calculate the R2 of 2ed moment
        R2_2ed_m_correct=1-LA.norm(diff_2ed_m_correct)**2/LA.norm(theoretical_2ed_m_correct)**2
        print(f'The R2 of 2ed moment for corrected estimator: {R2_2ed_m_correct}.')
        
        fig, axs = plt.subplots(2, 3, figsize=(18,5), dpi=100)
        #naive
        sns.heatmap(diff_B_naive, ax=axs[0, 0], cmap='viridis')      
        axs[0, 0].set_title('Naive Delta B')
        sns.heatmap(CR_naive, ax=axs[0, 1], cmap='PiYG') 
        axs[0, 1].set_title('Naive 95% CR')
        sns.heatmap(diff_2ed_m_naive, ax=axs[0,2], cmap='Spectral')
        axs[0,2].set_title('Niave Delta 2ed moment')
        #correct
        sns.heatmap(diff_B_correct, ax=axs[1, 0], cmap='viridis')      
        axs[1, 0].set_title('Corrected Delta B')
        sns.heatmap(CR_correct, ax=axs[1, 1], cmap='PiYG') 
        axs[1, 1].set_title('Corrected 95% CR')
        sns.heatmap(diff_2ed_m_correct, ax=axs[1,2], cmap='Spectral')
        axs[1,2].set_title('Corrected Delta 2ed moment')

        plt.show()

    elif case=="X*YV":
        
        sum_B=0
        vec_Bhat_array=np.zeros((dim_Y*dim_X,N_Simu))

        for simu_id in range(N_Simu):

              B_hat=single_simulation_accurate(case=case,dim_Y=dim_Y, dim_X=dim_X, N_Obs=N_Obs, 
                                          COV_X=Sigma_X, COV_U=Sigma_U, COV_Ex=Sigma_Ex, COV_Ey=Sigma_Ey, 
                                          COV_R=Sigma_R, COV_V=Sigma_V, B=B, A=A)
              
              sum_B+=B_hat
              vec_Bhat_array[:,simu_id]=B_hat.T.reshape(-1)
        #After simulation, calculate summarized results

        #plt.imshow(B)

        #1. biases of averaged average B_hat
        mean_B_hat=sum_B/N_Simu
        diff_B=mean_B_hat-B

        #calculate the R2 between \bar{\hat{B}} and true B
        R2_B=1-LA.norm(diff_B)**2/LA.norm(B)**2
        print(f'The $R^2$ on B is: {R2_B}.')
    
        #2. empirical cov
        vec_mean_B=mean_B_hat.T.reshape(-1)
        centered_vec_B=vec_Bhat_array-np.array([vec_mean_B,]*N_Simu).T
        empirical_cov=centered_vec_B@centered_vec_B.T/(N_Simu-1)
        vec_B=B.T.reshape(-1)
        Sigma_XsV=A@Sigma_V
        theoretical_cov=np.kron(LA.inv(Sigma_XsV)@Sigma_V@LA.inv(Sigma_XsV.T),(B@Sigma_X@B.T+Sigma_U))/N_Obs-np.outer(vec_mean_B, vec_mean_B)
        diff_cov=empirical_cov-theoretical_cov
        #calculate the R2 on cov
        R2_cov=1-LA.norm(diff_cov)**2/LA.norm(empirical_cov)**2
        print(f'The $R^2$ on covariance: {R2_cov}.')

        #4. calculate the CR% of 95% CI for each element of B
        vec_B=B.T.reshape(-1)
        vec_B_matrix=np.array([vec_B,]*N_Simu).T
        Var_B=np.diag(empirical_cov)
        sd_B_matrix=np.array([np.sqrt(Var_B),]*N_Simu).T
        lower_bound=vec_Bhat_array+ss.norm.ppf(0.025)*sd_B_matrix
        upper_bound=vec_Bhat_array+ss.norm.ppf(0.975)*sd_B_matrix
        InCI_Logical=np.greater(vec_B_matrix,lower_bound) & np.less(vec_B_matrix, upper_bound)
        InCI_numeric=InCI_Logical*1
        InCI=np.mean(InCI_numeric,axis=1)*100
        CR=InCI.reshape(dim_X, dim_Y).T
        #plt.imshow(CR)

        #3. calculate the 2ed moment matrix 
        empirical_2ed_m=vec_Bhat_array@vec_Bhat_array.T/N_Simu
        converg_valu_2=np.outer(vec_B,vec_B)
        theoretical_2ed_m=np.kron(LA.inv(Sigma_XsV)@Sigma_V@LA.inv(Sigma_XsV.T),(B@Sigma_X@B.T+Sigma_U))/N_Obs
        diff_2ed_m=empirical_2ed_m-theoretical_2ed_m
        #calculate the R2 of 2ed moment
        R2_2ed_m=1-LA.norm(diff_2ed_m)**2/LA.norm(empirical_2ed_m)**2
        print(f'The $R^2$ on 2ed moment: {R2_2ed_m}.')

        fig, axs = plt.subplots(2, 2, figsize=(18,5), dpi=100)
         
        sns.heatmap(diff_B, ax=axs[0, 0], cmap='PiYG') 
        axs[0, 0].set_title('Delta B')
        sns.heatmap(diff_cov, ax=axs[0, 1], cmap='PRGn')
        axs[0, 1].set_title('Difference covariance')
        sns.heatmap(CR, ax=axs[1,0], cmap='Spectral')
        axs[1,0].set_title('95% CR')
        sns.heatmap(diff_2ed_m, ax=axs[1, 1], cmap='BrBG')
        axs[1, 1].set_title('Difference 2ed moment')

        plt.show()




