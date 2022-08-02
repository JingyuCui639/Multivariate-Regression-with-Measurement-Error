import numpy as np
from numpy import linalg as LA 
import math
import time 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import pandas as pd
import plotly.graph_objects as go


def data_generation(case, dim_Y, dim_X, N_Obs,COV_X, COV_U, COV_Ex, COV_Ey, COV_R,
                    COV_V, B, A, X_distribution="Normal", diag_Ex_distribution="Normal",
                    U_distribution="Normal"):
    """
        case: string: "XY", "X*Y", "XY*", "X*Y*"
        dim_Y: dimension of response vector (Y_i)
        dim_X: dimension of covariates vector (X_i)
        N_Obs: the number of observations
        COV_X: the covariance matrix of the covariates (X_i)
        COV_U: the covariance matrix of the error (U_i)
        COV_EX: the covariance matrix of the measurement error (E_xi)
        X_distribution (string): "Normal", "Unifrom", "Laplace", "Gamma", "chi-square",("Cauchy","GMM"). indicating if the data X(covariates) and Ex(ME) generated from normal distributions
        diag_Ex_distribution (string): "Normal", "Unifrom", "Laplace", "Gamma", "chi-square",("Cauchy","GMM"). indicating the distribution from that ME Ex is generated when Sig_Ex is diagonal
        U_distribution (string): "Normal", "Unifrom", "Laplace", "Gamma", "chi-square",("Cauchy"). indicating the distribution that U is generated from 
    """


    #1. generate U from COV_U
    if U_distribution=="Normal":
        C_u=LA.cholesky(COV_U)
        Z=np.random.normal(loc=0.0, scale=1.0, size=(dim_Y,N_Obs))
        U=C_u@Z
    elif U_distribution=="Uniform":
        err_level=COV_U[0,0]
        U=np.random.uniform(low=-np.sqrt(3*err_level), high=np.sqrt(3*err_level), size=(dim_Y,N_Obs))
    elif U_distribution=="Cauchy":
        U=np.random.standard_cauchy(size=(dim_Y,N_Obs))
        err_level=COV_U[0,0]
        U=(U/U.std())*np.sqrt(err_level=COV_U[0,0])
    elif U_distribution=="Laplace":
        err_level=COV_U[0,0]
        U=np.random.laplace(loc=0, scale=np.sqrt(err_level/2), size=(dim_Y,N_Obs))
    elif U_distribution=="Gamma":
        Z=np.random.gamma(shape=4, scale=0.5, size=(dim_Y,N_Obs))
        Z=Z-2
        C_u=LA.cholesky(COV_U)
        U=C_u@Z
    elif U_distribution=="chi-square":
        Z=np.random.chisquare(2, size=(dim_Y,N_Obs))
        Z=(Z-2)/2
        C_u=LA.cholesky(COV_U)
        U=C_u@Z
        
    if case=="X*YV":

        C_v=LA.cholesky(COV_V)
        Z=np.random.normal(loc=0.0, scale=1.0, size=(dim_X,N_Obs))
        #Z=np.random.exponential(scale=0.5, size=(dim_X,N_Obs))
        V=C_v@Z 

        C_r=LA.cholesky(COV_R)
        #print(f"C_r shape: {C_r.shape}")
        Z=np.random.normal(loc=0.0, scale=1.0, size=(dim_X,N_Obs))
        #print(f"Z shape: {Z.shape}")
        R_err=C_r@Z

        X=A@V+R_err

        Y=B@X+U

        C_Ex=LA.cholesky(COV_Ex)
        Z=np.random.normal(loc=0.0, scale=1.0, size=(dim_X,N_Obs))
        Ex=C_Ex@Z
        X_star=X+Ex

        return Y, X, X_star, V

    #2. generate X_{q\times n} from COV_X
    if X_distribution=="Normal":
        C_x=LA.cholesky(COV_X)
        Z=np.random.normal(loc=0.0, scale=1.0, size=(dim_X, N_Obs))
        X=C_x@Z
    elif X_distribution=="GMM": #generate X based on full type of Sigma_X
        mean_list=[[-0.8]*dim_X, [0.2]*dim_X]
        var=COV_X-0.16
        var_list=[var, var]
        X=gaussian_mixture_generator(n=N_Obs, p=dim_X, n_comp=2, probs=[0.2,0.8], means=mean_list, covs=var_list)
    elif X_distribution=="Uniform":
        Z=np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=(dim_X, N_Obs))  
        C_x=LA.cholesky(COV_X)
        X=C_x@Z  
    elif X_distribution=="Cauchy":
        Z=np.random.standard_cauchy(size=(dim_X,N_Obs))
        Z=Z/Z.std()
        C_x=LA.cholesky(COV_X)
        X=C_x@Z
    elif X_distribution=="Laplace":
        Z=np.random.laplace(loc=0, scale=np.sqrt(0.5), size=(dim_X,N_Obs))
        C_x=LA.cholesky(COV_X)
        X=C_x@Z
    elif X_distribution=="chi-square":
        Z=np.random.chisquare(2, size=(dim_X,N_Obs))
        Z=(Z-2)/2
        C_x=LA.cholesky(COV_X)
        X=C_x@Z
    elif X_distribution=="Gamma":
        Z=np.random.gamma(shape=4, scale=0.5, size=(dim_X,N_Obs))
        Z=Z-2
        C_x=LA.cholesky(COV_X)
        X=C_x@Z
        
    #3. Construct Y
    Y=B@X+U
       
    if case=="X*Y":
        
        #4. generate measurement error 
        if diag_Ex_distribution=="Normal":
            C_Ex=LA.cholesky(COV_Ex)
            Z=np.random.normal(loc=0.0, scale=1.0,size=(dim_X,N_Obs))
            #Z=np.random.uniform(low=-2, high=3, size=(dim_X,N_Obs))
            #Z=np.random.exponential(scale=0.5, size=(dim_X,N_Obs))
            Ex=C_Ex@Z
        elif diag_Ex_distribution=="GMM":
            if COV_Ex[0,3]==0: #if COV_Ex is diagonal
                if COV_Ex[0,0]==0.2:
                    mean_list=[-0.8, 0.2]
                    cov_list=[0.04,0.04]
                elif COV_Ex[0,0]==0.5:
                    mean_list=[-1.2, 0.3]
                    cov_list=[0.14, 0.14]
                elif COV_Ex[0,0]==0.8:
                    mean_list=[-1.6, 0.4]
                    cov_list=[0.16, 0.16]
                random_data=gaussian_mixture_generator(n=N_Obs*dim_X, p=1, n_comp=2, probs=[0.2, 0.8], means=mean_list, covs=cov_list)
                Ex=random_data.reshape(dim_X, N_Obs)
                 
            else: # if COV_Ex is full matrix
                if COV_Ex[0,0]==0.2:
                    mean_list=[[-0.4]*dim_X, [0.1]*dim_X]
                    cov_list=[COV_Ex-0.04, COV_Ex-0.04]
                elif COV_Ex[0,0]==0.5:
                    mean_list=[[-0.6]*dim_X, [0.15]*dim_X]
                    cov_list=[COV_Ex-0.09, COV_Ex-0.09]
                elif COV_Ex[0,0]==0.8:
                    mean_list=[[-0.8]*dim_X, [0.2]*dim_X]
                    cov_list=[COV_Ex-0.16, COV_Ex-0.16]
                Ex=gaussian_mixture_generator(n=N_Obs, p=dim_X, n_comp=2, probs=[0.2,0.8], means=mean_list, covs=cov_list)           
        
        elif diag_Ex_distribution=="Uniform":
            Z=np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=(dim_X, N_Obs))  
            C_x=LA.cholesky(COV_Ex)
            Ex=C_x@Z
        elif diag_Ex_distribution=="Cauchy":
            Z=np.random.standard_cauchy(size=(dim_X,N_Obs))
            Z=Z/Z.std()
            C_x=LA.cholesky(COV_Ex)
            Ex=C_x@Z
        elif diag_Ex_distribution=="Laplace":
            Z=np.random.laplace(loc=0, scale=np.sqrt(0.5), size=(dim_X,N_Obs))
            C_x=LA.cholesky(COV_Ex)
            Ex=C_x@Z
        elif diag_Ex_distribution=="chi-square":
            Z=np.random.chisquare(2, size=(dim_X,N_Obs))
            Z=(Z-2)/2
            C_x=LA.cholesky(COV_Ex)
            Ex=C_x@Z
        elif diag_Ex_distribution=="Gamma":
            Z=np.random.gamma(shape=4, scale=0.5, size=(dim_X,N_Obs))
            Z=Z-2
            C_x=LA.cholesky(COV_Ex)
            Ex=C_x@Z
            
        #construct X*   
        X_star=X+Ex
        
        #return results
        return Y, X_star, X, U 




def cov_generator(dim, MODEL="random", random_sd=1, identity_sd=None, identity_var=None, random_seed=np.random.RandomState(99)):
    """
    Generate Covaraince matrix, either a random full positive definite matrix, or a diagonal matrix
    
    dim: an integer indicating the dimension of the covariance matrix.
    MODEL: a string indicating the model to generate the covariance: "random",
            "identity", "exp",...
    random_sd: for the "random" model, this sets the sd when first generate 
              N(0,1) rvs.
    identity_var: the var on the diagonal of the final matrix, should be
                 a list or an array of diagnal.    
    """
    identity_sd=np.array(identity_sd)
    
    if MODEL=="random":
        Z=random_seed.normal(0,random_sd,(dim,dim))
        W=0.5*(Z+Z.T)
        eig_val, _=LA.eig(W)
        Sigma=W+np.eye(dim)*math.ceil(abs(eig_val.min()))
        return Sigma
    elif MODEL=="identity":
        if identity_sd!=None:
            diag_values=identity_sd**2
        elif identity_var!=None:
            diag_values=identity_var
        if len(diag_values)==dim:
            Sigma=np.eye(dim)@np.diag(diag_values)
            return Sigma
        elif len(diag_values)==1:
            Sigma=np.eye(dim)*(diag_values)
            return Sigma        

        
def single_simulation_accurate(case,dim_Y, dim_X, N_Obs, COV_X, COV_U, COV_Ex, 
                            COV_Ey, COV_R, COV_V, B, A=0, X_distribution="Normal", diag_Ex_distribution="Normal",
                            U_distribution="Normal"):
    """
        case: "X*Y", "X*YV"
    """
    
    
    if case=="X*YV":

              Y, X, X_star, V=data_generation(case=case,dim_Y=dim_Y, dim_X=dim_X, N_Obs=N_Obs,
                            COV_X=COV_X, COV_U=COV_U, COV_Ex=COV_Ex,COV_Ey=COV_Ey, COV_R=COV_R,
                            COV_V=COV_V, B=B,A=A)
              
              #calculate the instrumental estimator
              S_YV=Y@V.T/N_Obs
              S_XV=X_star@V.T/N_Obs
              B_hat_inst=S_YV@LA.inv(S_XV)
              
              #calculate the naive estimator
              #empirical Sigma_YX
              S_YX=Y@X_star.T/N_Obs
              #empirical Sigma_X
              S_XX=X_star@X_star.T/N_Obs
              #Estimating B
              B_hat_naive=S_YX@LA.inv(S_XX)
              
              #calculate the LSE
              S_X=X@X.T/N_Obs
              S_yx=Y@X.T/N_Obs
              B_hat_LSE=S_yx@LA.inv(S_X)
              
              l=list(B_hat_LSE.T.reshape(-1))+list(B_hat_naive.T.reshape(-1))+list(B_hat_inst.T.reshape(-1))
              return l
              
    else:

              #Step 2. data generation

              Y_data,X_surrogate, X_true, _=data_generation(case=case,dim_Y=dim_Y, dim_X=dim_X, N_Obs=N_Obs,
                                          COV_X=COV_X, COV_U=COV_U, COV_Ex=COV_Ex,COV_Ey=COV_Ey, COV_R=COV_R,
                                          COV_V=COV_V, B=B,A=A, X_distribution=X_distribution, diag_Ex_distribution=diag_Ex_distribution,
                                          U_distribution=U_distribution)
                            
              # calculate the B har when the true X is available
              S_X=X_true@X_true.T/N_Obs
              S_yx=Y_data@X_true.T/N_Obs
              B_hat_trueX=S_yx@LA.inv(S_X)
              
              #Step 3. calculate the naive estimator
              #empirical Sigma_YX
              S_YX=Y_data@X_surrogate.T/N_Obs
              #empirical Sigma_X
              S_XX=X_surrogate@X_surrogate.T/N_Obs
              #Estimating B
              B_hat_naive=S_YX@LA.inv(S_XX)

              #Step 4. Calculate the corrected estimator
              K1=(COV_X+COV_Ex)@LA.inv(COV_X)
              B_hat_correct_trueCov=B_hat_naive@K1
              
              # return the three estimates
              l=list(B_hat_trueX.T.reshape(-1))+list(B_hat_naive.T.reshape(-1))+list(B_hat_correct_trueCov.T.reshape(-1))#+list(B_hat_correct_estCov.T.reshape(-1))
              return l
def main_simu_func(case, dim_Y, dim_X, cov_type, N_Obs, error_level, 
                   change="Sigma_X", Sigma_Ex_type="diagonal", 
                   X_distribution="Normal", diag_Ex_distribution="Normal", U_distribution="Normal"):
    """_summary_

    Args:
        case (_type_): _description_
        dim_Y (_type_): _description_
        dim_X (_type_): _description_
        cov_type (_type_): _description_
        N_Obs (_type_): _description_
        error_level (_type_): _description_
        change (str, optional): "Sigma_X" or "Sigma_Ex", indicates which wil be changed for difference scenarios.
        Sigma_Ex_type (str, optional): 1. "full and proportional"; 2. "full and not proportional";
                                        3. "sparse"; 4. "diagonal", indicating the type of covariance structure of
                                        measurement errors

    Returns:
        _type_: _description_
    """
    start_time=time.time()
     
    # #obtain the inputs
    # case=sys.argv[1]
    # dim_Y=int(sys.argv[2])
    # dim_X=int(sys.argv[3])
    
    #error_level=0.8
    
    #parameters change in 2 cases
    #N_Obs=1000 # n (# of columns) in the data X and Y: 100, 1000, 10000, 100000
    #parmtes fixed in 2 cases
    N_Simu=100000 # simulation number to calculate the variance of B.hat
    #N_simu_cov=100 # number of simulation in order to calculate the mean of estimated cov for B.hat
    N_est_simu=100000 # the number fo repetition to estimat the E (S_{XX*} and S_{X^*X^*})
    
    ####################################################
    # Step 1. define B, Sigma_X, A and other parameters#
    ####################################################
    if dim_Y==4 and dim_X==7:

        #B=np.random.uniform(low=-5, high=5, size=(dim_Y, dim_X))
        B=np.array([[-4.94, -2.35, 3.60, 3.87, -4.14, 3.01, 0.35],
                    [3.77, -4.28, 1.30, -4.82, -3.45, -0.70, -2.64],
                    [0.24, -3.10, 4.06, -0.02, -1.90, 2.80, 1.07],
                    [-4.71, -1.93, 1.21, 4.98, -1.34, 4.19, -1.17]])
        
        if cov_type=="full":
            
            #Case: (4,7) full matrix
            Sigma_X=np.array([[1,     0.3, 0.1, 0.05, 0.1, -0.06, 0.1],
                            [0.3,   1,   0.3, 0.1, -0.1, -0.1,  0.3],
                            [0.1,   0.3, 1,   0.3,  0.1,  0.1, -0.1],
                            [0.05,  0.1, 0.3, 1,    0.3,  0.1,  0.05],
                            [0.1,  -0.1, 0.1, 0.3,  1,    0.3,  0.1],
                            [-0.06,-0.1, 0.1, 0.1,  0.3,  1,    0.3],
                            [0.1,   0.3,-0.1, 0.05, 0.1,  0.3,   1]])
        elif cov_type=="sparse":

            #Case: (4,7) sparse matrix
            Sigma_X=np.array([[1,     0.3, 0.1, 0.05,  0,   0,    0],
                            [0.3,   1,    0,   0,    0,   0,    0],
                            [0.1,   0,    1,   0,    0,   0,    0],
                            [0.05,  0,    0,   1,    0.3, 0.1,  0.05],
                            [0,     0,    0,   0.3,  1,   0,    0],
                            [0,     0,    0,   0.1,  0,   1,    0],
                            [0,     0,    0,   0.05, 0,   0,    1]])
            
        elif cov_type=="identity":

            #Case: (4,7) diagonal matrix
            Sigma_X=np.array([[1,   0, 0, 0, 0, 0, 0],
                            [0,   1, 0, 0, 0, 0, 0],
                            [0,   0, 1, 0, 0, 0, 0],
                            [0,   0, 0, 1, 0, 0, 0],
                            [0,   0, 0, 0, 1, 0, 0],
                            [0,   0, 0, 0, 0, 1, 0],
                            [0,   0, 0, 0, 0, 0, 1]])  
            
    elif dim_Y==3 and dim_X==3:
           
        B=np.array([[ 2.5, 3, 2.5],
                    [ 0.5, -3.2,-3.8 ],
                    [-0.3, 3.5, 4.8]])
        
        if cov_type=="full":
                     
            #Case: (3,3) full matrix
            Sigma_X=np.array([[1, 0.5, 0.1],
                            [0.5, 1, 0.5],
                            [0.1, 0.5, 1]])
        elif cov_type=="sparse":
               
            #Case: (3,3) sparse matrix
            Sigma_X=np.array([[1,  0, 0.5],
                            [0,  1,  0],
                            [0.5,0,  1]])
        elif cov_type=="identity":
    
            #Case: (3,3) diagonal matrix
            Sigma_X=np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
            
    print(f"Matrix B:{B}")
       
    # #plot B matrix (heatmap)
    # fig, axs = plt.subplots(1,1, figsize=(18,5), dpi=100)        
    # sns.heatmap(B, cmap='coolwarm',center=0,annot=True) 
    # axs.set_title('True B')
    # plt.show()
    
    print(f"Sigma_X: {Sigma_X}")
    
    # #plot Sigma_XX matrix (heatmap)
    # fig, axs = plt.subplots(1,1, figsize=(18,5), dpi=100)        
    # sns.heatmap(Sigma_X, cmap='coolwarm',center=0,annot=True) 
    # axs.set_title('Sigma_X')
    # plt.show()

    
    A=np.array([[ 0.91,0.49,1.00,0.73,0.25,0.80,0.29],
                [ 0.76,  1.68,  0.83,  1.35, -0.31, 0.14,1.38],
                [ 0.38,0.01,1.05,1.52,0.32,0.05, 1.96],
                [ 1.37,0.17,1.17,1.93,1.65,1, 1.63],
                [ 1.69,0.54,1.07,1.44,0.45,-0.09,0.04],
                [ 1.92,1,0.73,0.47, 0.98,1.66, 0.64],
                [-0.48, -0.28, -0.31,  0.05, -0.11,  1.42,0.89 ]])
    
    
    #print(f"Matrix A:{A}")

    Sigma_V=Sigma_X #cov_generator(dim=dim_X, MODEL="random")

    Sigma_U=cov_generator(dim=dim_Y, MODEL="identity", identity_sd=[0.1])

    if case=="XY":

        Sigma_Ex=0
        Sigma_Ey=0
        Sigma_R=0
    
    elif case=="X*Y":

        if change=="Sigma_X":
            Sigma_Ex=cov_generator(dim=dim_X, MODEL="identity", identity_var=[error_level])
        elif change=="Sigma_Ex":
            if Sigma_Ex_type=="full and proportional":
                Sigma_Ex=error_level*Sigma_X
            elif Sigma_Ex_type=="full and not proportional":
                Sigma_Ex=error_level*np.array([[ 0.5 , 0.25, 0.15, 0.05, 0.025, -0.05, -0.15 ],
                                    [ 0.25, 0.5, 0.125, 0.15, 0.05, 0.025, -0.05 ],
                                    [ 0.15, 0.125, 0.5,  0.25, 0.15, 0.05, 0.025],
                                    [ 0.05, 0.15,  0.25, 0.5,  0.125,0.15, 0.05 ],
                                    [ 0.025,0.05, 0.15, 0.125, 0.5, 0.25, 0.15 ],
                                    [-0.05, 0.025, 0.05, 0.15,  0.25, 0.5 , 0.125],
                                    [-0.15, -0.05,  0.025, 0.05,  0.15, 0.125,  0.5]])/0.5
            elif Sigma_Ex_type=="sparse":
                Sigma_Ex=np.array([[ 0.5 , 0.25, 0.15, 0.05, 0, 0, 0 ],
                                   [ 0.25, 0.5, 0, 0, 0, 0, 0],
                                   [ 0.15,0, 0.5, 0, 0, 0, 0],
                                   [ 0.05, 0,  0, 0.5,  0.125,0.15, 0.05 ],
                                   [ 0,0, 0, 0.125, 0.5, 0, 0 ],
                                   [ 0, 0, 0, 0.15,  0, 0.5 , 0],
                                   [ 0,0,  0, 0.05,  0, 0,  0.5]])
            elif Sigma_Ex_type=="diagonal":
                Sigma_Ex=error_level*np.eye(dim_X)
                
        print(f"Sigma_Ex={Sigma_Ex}")
        
        Sigma_Ey=0
        Sigma_R=cov_generator(dim=dim_Y, MODEL="identity", identity_sd=[0.1])
        
        K=(Sigma_X+Sigma_Ex)@LA.inv(Sigma_X)

    elif case=="X*YV":
        
        if change=="Sigma_X":
            Sigma_Ex=cov_generator(dim=dim_X, MODEL="identity", identity_var=[error_level])
        elif change=="Sigma_Ex":
            if Sigma_Ex_type=="full and proportional":
                Sigma_Ex=0.5*Sigma_X
            elif Sigma_Ex_type=="full and not proportional":
                # define a covariance matrix for the measurement error
                Sigma_Ex=np.array([[ 0.5 , 0.25, 0.15, 0.05, 0.025, -0.05, -0.15 ],
                                    [ 0.25, 0.5, 0.125, 0.15, 0.05, 0.025, -0.05 ],
                                    [ 0.15, 0.125, 0.5,  0.25, 0.15, 0.05, 0.025],
                                    [ 0.05, 0.15,  0.25, 0.5,  0.125,0.15, 0.05 ],
                                    [ 0.025,0.05, 0.15, 0.125, 0.5, 0.25, 0.15 ],
                                    [-0.05, 0.025, 0.05, 0.15,  0.25, 0.5 , 0.125],
                                    [-0.15, -0.05,  0.025, 0.05,  0.15, 0.125,  0.5]])
            elif Sigma_Ex_type=="sparse":
                
                Sigma_Ex=np.array([[ 0.5 , 0.25, 0.15, 0.05, 0, 0, 0 ],
                                   [ 0.25, 0.5, 0, 0, 0, 0, 0],
                                   [ 0.15,0, 0.5, 0, 0, 0, 0],
                                   [ 0.05, 0,  0, 0.5,  0.125,0.15, 0.05 ],
                                   [ 0,0, 0, 0.125, 0.5, 0, 0 ],
                                   [ 0, 0, 0, 0.15,  0, 0.5 , 0],
                                   [ 0,0,  0, 0.05,  0, 0,  0.5]]) 
            elif Sigma_Ex_type=="diagonal":
                Sigma_Ex=0.5*np.diag(Sigma_X)*np.eye(dim_X)
                
        print(f"Sigma_Ex={Sigma_Ex}")
        
        Sigma_Ey=0
        Sigma_R=cov_generator(dim=dim_X, MODEL="identity", identity_sd=[0.1])
        Sigma_X=A@Sigma_V@A.T+Sigma_R
        
        print(f"Sigma_X={Sigma_X}")

    #######################
    # Step 2. simulations #
    #######################
    if case== "X*Y":
        
    ########### paralell programing    
    # empirical_cov_naive=np.zeros((dim_X*dim_Y, dim_X*dim_Y))
    # empirical_cov_correct=np.zeros((dim_X*dim_Y, dim_X*dim_Y))
    # for m in range(N_simu_cov):      
        # ##Init multiprocessing.Pool()
        # pool = mp.Pool(mp.cpu_count())
        # #vec_B_naive = []
        # #vec_B_correct=[]
        
        # np.random.seed(1)
        # # call apply_async() without callback
        # result_objects = [pool.apply_async(single_simulation_accurate, args=(case,dim_Y, dim_X, N_Obs,
        #                                                                         Sigma_X, Sigma_U, Sigma_Ex, Sigma_Ey,
        #                                                                         Sigma_R, Sigma_V, B)) for simu_id in range(N_Simu)]

        # # result_objects is a list of pool.ApplyResult objects
        
        # results = [r.get() for r in result_objects]

        # #vec_B_correct=[r.get()[1] for r in result_objects]

        # pool.close()
        # pool.join()
        
        np.random.seed(10) #FIX the seed as 10
        
        result_list=[]
        
        for simu_id in range(N_Simu):    
            
            result_list.append(single_simulation_accurate(case, dim_Y, dim_X, N_Obs, Sigma_X, Sigma_U, Sigma_Ex, Sigma_Ey,Sigma_R, 
                                                          Sigma_V, B,0, X_distribution, diag_Ex_distribution, U_distribution))
                      
            
        results_arr=np.array(result_list)
        print(f"shape of results_arr: {results_arr.shape}")
        
        #the estimate when true X is observed
        vec_Bhat_trueX=results_arr[:,:dim_X*dim_Y]
        vec_Bhat_array_trueX=vec_Bhat_trueX.T #dim: q by N_Obs
        print(f"shape of vec_B_trueX:{vec_Bhat_array_trueX.shape}")
         
        #the naive estimate under X*   
        vec_B_naive=results_arr[:,dim_X*dim_Y:2*dim_X*dim_Y]
        vec_Bhat_array_naive=vec_B_naive.T
        print(f"shape of vec_Bhat_array_naive:{vec_Bhat_array_naive.shape}")

        #the corrected estimate under X* using true covariance matrix 
        vec_B_correct=results_arr[:,2*dim_X*dim_Y:3*dim_X*dim_Y]
        vec_Bhat_array_correct=vec_B_correct.T
        print(f"shape of vec_Bhat_array_correct:{vec_Bhat_array_correct.shape}")            
        
        ########################################################
        ### Evaluating for B hat
        #########################################################       
          
        #1. biases of averaged B_hat
        
        #B hat under true X, over simulations and elements
        vec_mean_B_hat_trueX=vec_Bhat_array_trueX.mean(axis=1)
        matrix_mean_B_hat_trueX=vec_mean_B_hat_trueX.reshape(dim_X,dim_Y).T
        avgBias_diff_B_trueX=(matrix_mean_B_hat_trueX-B).mean()
        print(f"avgBias of Bhat (LSE):{avgBias_diff_B_trueX}")

        #naive ,over simulations and elements
        vec_mean_B_hat_naive=vec_Bhat_array_naive.mean(axis=1)
        matrix_mean_B_hat_naive=vec_mean_B_hat_naive.reshape(dim_X,dim_Y).T
        avgBias_diff_B_naive=(matrix_mean_B_hat_naive-B).mean()
        print(f"avgBias of Bhat (naive):{avgBias_diff_B_naive}")
        
        # #correct with true cov ,over simulations and elements
        vec_mean_B_hat_correct=vec_Bhat_array_correct.mean(axis=1)
        matrix_mean_B_hat_correct=vec_mean_B_hat_correct.reshape(dim_X,dim_Y).T
        avgBias_diff_B_correct=(matrix_mean_B_hat_correct-B).mean()
        print(f"avgBias of Bhat (crrect true cov):{avgBias_diff_B_correct}")
        
        ########################################################################################
        #plot the violin plot of true B, LSE, Bhat_naive, Bhat_correct for averaged estimates
        ###########################################################################################
        estimate_data=pd.DataFrame(np.concatenate((                  
                            matrix_mean_B_hat_naive.reshape(-1,1),
                            matrix_mean_B_hat_correct.reshape(-1,1),
                            matrix_mean_B_hat_trueX.reshape(-1,1)),axis=1), columns=[
                                                                "Naive Estimate", 
                                                                "Corrected Estimate",
                                                                "LSE"])
        

        layout = go.Layout(
            autosize=False,
            width=1000,
            height=600,)

        fig = go.Figure( layout=layout)

        days = ["Naive Estimate", "Corrected Estimate", "LSE"]


        for day in days:
            fig.add_trace(go.Violin(x=estimate_data[day],
                                    name=day,
                                    box_visible=True,
                                    meanline_visible=False
                                    #,points="all"
                                    ))
        fig.show()
        
        ###################################################################################
        #ploting the heatmap for the naive and corrected estimates for averaged estimates #
        ###################################################################################
        min_legend=B.min()
        max_legend=B.max()
        fig, axs = plt.subplots(3,1,figsize=(4,4), dpi=100)       
        sns.heatmap(matrix_mean_B_hat_trueX, ax=axs[0], cmap='Spectral',center=0,vmin=min_legend, vmax=max_legend,annot=False)
        axs[0].set_title('$\hat{B}$')
        sns.heatmap(matrix_mean_B_hat_correct, ax=axs[1], cmap='Spectral',center=0,vmin=min_legend, vmax=max_legend,annot=False) #PiYG #icefire
        axs[1].set_title('$\hat{B}_{x^*c}$')
        sns.heatmap(matrix_mean_B_hat_naive, ax=axs[2], cmap='Spectral',center=0, vmin=min_legend, vmax=max_legend,annot=False)  #gist_rainbow    
        axs[2].set_title('$\hat{B}_{x^*}$')
            # sns.heatmap(diff_B_correct, ax=axs[1,0], cmap='PiYG',center=0,annot=True)
            # axs[1,0].set_title('$\hat{B}_{x^*c1}-B$')
            # sns.heatmap(Bhat_correct_estCov, ax=axs[1,1], cmap='PiYG', center=0, annot=True)
            # axs[1,1].set_title('$\hat{B}_{x^*c2}-B$')
        fig.tight_layout()
        plt.show()
        
        #######################################################################
        # ploting the 3D-scatter plot for all estimators for averaged estimates
        ########################################################################
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(20,20), dpi=125)
        X = np.arange(0, dim_X, 1)
        Y = np.arange(0, dim_Y, 1)
        Y, X = np.meshgrid(Y, X)
        Z_trueB=B[Y,X]
        Z_B_lse=matrix_mean_B_hat_trueX[Y,X]
        Z_naive_Bhat=matrix_mean_B_hat_naive[Y,X]
        Z_correct_Bhat=matrix_mean_B_hat_correct[Y,X]
        Z4=np.zeros((dim_Y, dim_X))[Y,X]
        # Plot the surface.
        #ax.plot_surface(X, Y, Z_B, linewidth=0, antialiased=False, alpha=0.1, color="blue")
        #ax.plot_surface(X, Y, Z_Bxerr, linewidth=0, antialiased=False, alpha=0.5, color="red")
        ax.scatter(Y, X, Z_trueB, linewidth=0, antialiased=False, alpha=0.9, marker='o', color="blue", s=120, label="$B$")
        ax.scatter(Y, X, Z_B_lse, linewidth=0, antialiased=False, alpha=1, marker='s',color="yellow",  s=60, label="$\hat{B}$")
        ax.scatter(Y, X, Z_correct_Bhat, linewidth=0, antialiased=False, alpha=1, marker="^", color="red", s=50,  label="$\hat{B}_{x^*c}$")
        ax.scatter(Y, X, Z_naive_Bhat, linewidth=0, antialiased=False, alpha=1, marker="X", color="green", s=90,  label="$\hat{B}_{x^*}$")
        #ax.scatter(Y, X, Z_correct_Bhat_estCov, linewidth=0, antialiased=False, alpha=0.8, marker='D', s=20, label="$\hat{B}_{x^*c2}$")
        ax.plot_surface(Y, X, Z4, linewidth=0, antialiased=False, alpha=0.5, color="grey")
        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=10)
        ax.set_xlabel('p')
        ax.set_ylabel('q')
        ax.set_zlabel(' ')
        ax.legend(loc=1)
        plt.show()
        
        ##################################
        ### avg Empirical standard errors for B hat
        ##################################
        
        #2. empirical cov of B hat's
 
        #calculate the empirical covariance matrix for the B hat under true X
        centered_vec_Bhat_trueX=vec_Bhat_array_trueX-np.array([vec_mean_B_hat_trueX,]*N_Simu).T
        empirical_cov_trueX=centered_vec_Bhat_trueX@centered_vec_Bhat_trueX.T/(N_Simu-1)
        Bhat_LSE_avgSEE=np.sqrt(empirical_cov_trueX.diagonal()).mean()
        print(f"The avgSEE (LSE): {Bhat_LSE_avgSEE}")
        
        
        #Let mean estimate as the center to calculate the empirical variance
        centered_vec_B_naive=vec_Bhat_array_naive-np.array([vec_mean_B_hat_naive,]*N_Simu).T
        empirical_cov_naive=centered_vec_B_naive@centered_vec_B_naive.T/(N_Simu-1)
        Bhat_naive_avgSEE=np.sqrt(empirical_cov_naive.diagonal()).mean()
        print(f"The avgSEE (naive): {Bhat_naive_avgSEE}")

      
        #Let mean estimate as the center to calculate the empirical variance
        centered_vec_B_correct=vec_Bhat_array_correct-np.array([vec_mean_B_hat_correct,]*N_Simu).T
        empirical_cov_correct=centered_vec_B_correct@centered_vec_B_correct.T/(N_Simu-1)
        Bhat_correct_avgSEE=np.sqrt(empirical_cov_correct.diagonal()).mean()
        print(f"The avgSEE (correct with true cov): {Bhat_correct_avgSEE}")

        # theoretical cov of estimators 
         
        if case=="X*Y":
            ##referenced term cov naive
            temp2=B@Sigma_X@LA.inv(Sigma_X+Sigma_Ex)
            temp2T=temp2.T
            
            # generate estimation data
            vec_vecT=np.zeros((dim_X**2, dim_X**2))
            
            for i in range(N_est_simu):
                
                _, X_star, X, _=data_generation(case=case,dim_Y=dim_Y, dim_X=dim_X, N_Obs=N_Obs,
                                                COV_X=Sigma_X, COV_U=Sigma_U, COV_Ex=Sigma_Ex,COV_Ey=Sigma_Ey, COV_R=Sigma_R,
                                                COV_V=Sigma_V, B=B,A=A,  X_distribution=X_distribution, U_distribution=U_distribution)
                temp=X@X_star.T@LA.inv(X_star@X_star.T)
                tempT=temp.T
                vec_tempT=tempT.reshape(-1,1)
                vec_vecT+=vec_tempT@tempT.reshape(1,-1)                
                
            mean_vec_vecT=vec_vecT/N_est_simu

            #theoretical cov for B LSE
            theoretical_cov_trueX=np.kron(LA.inv(Sigma_X), Sigma_U)/N_Obs
            Bhat_LSE_avgSEM=np.sqrt(theoretical_cov_trueX.diagonal()).mean()
            print(f"The avgSEM for B hat (LSE): {Bhat_LSE_avgSEM}")
            # diff_cov_trueX=empirical_cov_trueX-theoretical_cov_trueX
            # Cov_LSE_avgBias=np.mean(diff_cov_trueX)
            # print(f"avgBias for diff cov (LSE):{Cov_LSE_avgBias}")
            # Cov_LSE_Frob=LA.norm(diff_cov_trueX)
            # print(f"L2-norm of diff_cov_trueX (LSE): {Cov_LSE_Frob}")
            # Cov_LSE_infinity_norm=abs(diff_cov_trueX).max()
            # print(f" The infinity norm of diff cov for B hat (LSE):{Cov_LSE_infinity_norm}")
                        
            #theoretical cov for naive estimate
            theoretical_cov_naive2=np.kron(LA.inv(Sigma_X+Sigma_Ex), Sigma_U)/N_Obs\
                                    +np.kron(np.eye(dim_X), B)@mean_vec_vecT@np.kron(np.eye(dim_X), B.T)\
                                    -temp2T.reshape(-1,1)@temp2T.reshape(1,-1)
            #diff_cov_naive_refe2=empirical_cov_naive-theoretical_cov_naive2
            print(f"diagonals in theoretical cov of B hat (naive):{theoretical_cov_naive2.diagonal()}")
            
            Bhat_naive_avgSEM=np.sqrt(theoretical_cov_naive2.diagonal()).mean()
            print(f"The avgSEM for B hat (naive): {Bhat_naive_avgSEM}")
            # Cov_naive_avgBias=np.mean(diff_cov_naive_refe2)
            # print(f"avgBias for diff cov (naive):{Cov_naive_avgBias}")
            # Cov_naive_Frob=LA.norm(diff_cov_naive_refe2)
            # print(f"L2-norm of the diff_cov_naive_refe2 (naive): {Cov_naive_Frob}")
            # Cov_naive_infinity_norm=abs(diff_cov_naive_refe2).max()
            # print(f"The infinity norm of the diff cov (naive): {Cov_naive_infinity_norm}")
                      
            
            #theoretical cov for corrected estimate
            theoretical_cov_corrected=np.kron(LA.inv(Sigma_X)@(Sigma_X+Sigma_Ex)@LA.inv(Sigma_X), Sigma_U)/N_Obs\
                                        +np.kron(K.T, B)@mean_vec_vecT@np.kron(K, B.T)\
                                        -B.T.reshape(-1,1)@B.T.reshape(1,-1)
            #diff_cov_correct=empirical_cov_correct-theoretical_cov_corrected
            print(f"diagonals of the theoretical cov of B hat (correct with true cov):{theoretical_cov_corrected.diagonal()}")
            Bhat_correct_avgSEM=np.sqrt(theoretical_cov_corrected.diagonal()).mean()
            print(f"The avgSEM for B hat (corrected with true cov):{Bhat_correct_avgSEM}")
            # Cov_correct_avgBias=np.mean(diff_cov_correct)
            # print(f"avgBias for diff cov (correct with true cov):{Cov_correct_avgBias}")
            # Cov_correct_Frob=LA.norm(diff_cov_correct)
            # print(f"L2-norm of diff_cov (correct with true cov): {Cov_correct_Frob}")
            # Cov_correct_infinity_norm=abs(theoretical_cov_corrected).max()
            # print(f" Infinite norm of diff cov (correct with true cov): {Cov_correct_infinity_norm}")
            
            #4. calculate the CR% of 95% CI for each element of B
            vec_B=B.T.reshape(-1)
            vec_B_matrix=np.array([vec_B,]*N_Simu).T
            
            #LSE
            Var_B_LSE=np.diag(theoretical_cov_trueX)
            sd_B_LSE=np.array([np.sqrt(Var_B_LSE),]*N_Simu).T
            lower_bound_LSE=vec_Bhat_array_trueX+ss.norm.ppf(0.025)*sd_B_LSE
            upper_bound_LSE=vec_Bhat_array_trueX+ss.norm.ppf(0.975)*sd_B_LSE
            InCI_logical_LSE=np.greater(vec_B_matrix, lower_bound_LSE) & np.less(vec_B_matrix,upper_bound_LSE)
            InCI_numeric_LSE=InCI_logical_LSE*1
            In_LSE=np.mean(InCI_numeric_LSE, axis=1)*100
            Bhat_LSE_avgCR=np.mean(In_LSE)
            print(f"The avgCI 95% for Bhat (LSE):{Bhat_LSE_avgCR}")
        
            #naive case
            Var_B_naive=np.diag(theoretical_cov_naive2)
            sd_B_matrix_naive=np.array([np.sqrt(Var_B_naive),]*N_Simu).T
            lower_bound_naive=vec_Bhat_array_naive+ss.norm.ppf(0.025)*sd_B_matrix_naive
            upper_bound_naive=vec_Bhat_array_naive+ss.norm.ppf(0.975)*sd_B_matrix_naive
            InCI_Logical_naive=np.greater(vec_B_matrix,lower_bound_naive) & np.less(vec_B_matrix, upper_bound_naive)
            InCI_numeric_naive=InCI_Logical_naive*1
            InCI_naive=np.mean(InCI_numeric_naive,axis=1)*100
            #CR_naive=InCI_naive.reshape(dim_X, dim_Y).T
            Bhat_naive_avgCR=np.mean(InCI_naive)
            print(f"The avgCI 95% for Bhat (naive): {Bhat_naive_avgCR}")
            #correct
            Var_B_correct=np.diag(theoretical_cov_corrected)
            sd_B_matrix_correct=np.array([np.sqrt(Var_B_correct),]*N_Simu).T
            lower_bound_correct=vec_Bhat_array_correct+ss.norm.ppf(0.025)*sd_B_matrix_correct
            upper_bound_correct=vec_Bhat_array_correct+ss.norm.ppf(0.975)*sd_B_matrix_correct
            InCI_Logical_correct=np.greater(vec_B_matrix, lower_bound_correct) & np.less(vec_B_matrix,upper_bound_correct)
            InCI_numeric_correct=InCI_Logical_correct*1
            InCI_correct=np.mean(InCI_numeric_correct, axis=1)*100
            #CR_correct=InCI_correct.reshape(dim_X, dim_Y).T
            Bhat_correct_avgCR=np.mean(InCI_correct)
            print(f"The avgCI 95% for Bhat (correct with true cov):{Bhat_correct_avgCR}")

                
            end_time=time.time()
            print(f"time used: {(end_time-start_time)/60} mins")
        
            return np.array([avgBias_diff_B_trueX,Bhat_LSE_avgSEE,Bhat_LSE_avgSEM,Bhat_LSE_avgCR,
                         avgBias_diff_B_naive,Bhat_naive_avgSEE,Bhat_naive_avgSEM,Bhat_naive_avgCR,	
                         avgBias_diff_B_correct,Bhat_correct_avgSEE,Bhat_correct_avgSEM,Bhat_correct_avgCR
                        #  avgBias_diff_B_correct_estCov,	Bhat_correct2_avgSEE,Bhat_correct2_Frob,Bhat_correct2_infinity_norm,
                        #  Cov_LSE_avgBias,Cov_LSE_Frob,Cov_LSE_infinity_norm,	
                        #  Cov_naive_avgBias,	Cov_naive_Frob,	Cov_naive_infinity_norm,	
                        #  Cov_correct_avgBias,Cov_correct_Frob,Cov_correct_infinity_norm
                          ])
            
    elif case=="X*YV":
        
        np.random.seed(1) #FIX the seed as 1
        
        result_list=[]
        
        for simu_id in range(N_Simu):    
            
            result_list.append(single_simulation_accurate(case, dim_Y, dim_X, N_Obs, None, Sigma_U, Sigma_Ex, Sigma_Ey,Sigma_R, Sigma_V, B,A))
                                                                                       
        results_arr=np.array(result_list)
        print(f"shape of results_arr: {results_arr.shape}")
        
        #################################
        #calculate the averaged estimates
        #################################
        #the LSE estimates
        vec_Bhat_trueX=results_arr[:,:dim_X*dim_Y]
        vec_Bhat_array_trueX=vec_Bhat_trueX.T #dim: q by N_Obs
        print(f"shape of vec_B_trueX:{vec_Bhat_array_trueX.shape}")
        #B hat under true X, over simulations and elements
        vec_mean_B_hat_trueX=vec_Bhat_array_trueX.mean(axis=1)
        matrix_mean_B_hat_trueX=vec_mean_B_hat_trueX.reshape(dim_X,dim_Y).T
        avgBias_diff_B_trueX=(matrix_mean_B_hat_trueX-B).mean()
        print(f"avgBias of Bhat (LSE):{avgBias_diff_B_trueX}")
         
        #the naive estimate under X*   
        vec_B_naive=results_arr[:,dim_X*dim_Y:2*dim_X*dim_Y]
        vec_Bhat_array_naive=vec_B_naive.T
        print(f"shape of vec_Bhat_array_naive:{vec_Bhat_array_naive.shape}")
        #naive, over simulations and elements
        vec_mean_B_hat_naive=vec_Bhat_array_naive.mean(axis=1)
        matrix_mean_B_hat_naive=vec_mean_B_hat_naive.reshape(dim_X,dim_Y).T
        avgBias_diff_B_naive=(matrix_mean_B_hat_naive-B).mean()
        print(f"avgBias of Bhat (naive):{avgBias_diff_B_naive}")

        #the corrected estimate under X* using true covariance matrix 
        vec_B_inst=results_arr[:,2*dim_X*dim_Y:3*dim_X*dim_Y]
        vec_Bhat_array_inst=vec_B_inst.T
        print(f"shape of vec_Bhat_array_inst:{vec_Bhat_array_inst.shape}")
        # #correct with true cov ,over simulations and elements
        vec_mean_B_hat_inst=vec_Bhat_array_inst.mean(axis=1)
        matrix_mean_B_hat_inst=vec_mean_B_hat_inst.reshape(dim_X,dim_Y).T
        avgBias_diff_B_inst=(matrix_mean_B_hat_inst-B).mean()
        print(f"avgBias of Bhat (instrumental variables):{avgBias_diff_B_inst}")
        
        #plot the violin plot of true B, LSE, Bhat_naive, Bhat_correct for the averaged estimates
        estimate_data=pd.DataFrame(np.concatenate(( 
                            matrix_mean_B_hat_naive.reshape(-1,1),
                            matrix_mean_B_hat_inst.reshape(-1,1),
                            matrix_mean_B_hat_trueX.reshape(-1,1)),axis=1), columns=["Naive Estimate",
                                                                "IV Estimate", 
                                                                "LSE"])
        
        layout = go.Layout(
            autosize=False,
            width=1000,
            height=600,)

        fig = go.Figure( layout=layout)

        days = [ "Naive Estimate", "IV Estimate","LSE"]


        for day in days:
            fig.add_trace(go.Violin(x=estimate_data[day],
                                    name=day,
                                    box_visible=True,
                                    meanline_visible=False
                                    #,points="all"
                                    ))
        fig.show()
        
        #ploting the heatmap for the naive and IV estimates for averaged results
        min_legend=-5
        max_legend=5
        fig, axs = plt.subplots(3,1,figsize=(4,4), dpi=100)
        sns.heatmap(matrix_mean_B_hat_trueX, ax=axs[0], cmap='Spectral',center=0,vmin=min_legend, vmax=max_legend,annot=False)
        axs[0].set_title('$\hat{B}$')
        sns.heatmap(matrix_mean_B_hat_inst, ax=axs[1], cmap='Spectral',center=0,vmin=min_legend, vmax=max_legend,annot=False) #PiYG #icefire
        axs[1].set_title('$\hat{B}_{V}$')
        sns.heatmap(matrix_mean_B_hat_naive, ax=axs[2], cmap='Spectral',center=0, vmin=min_legend, vmax=max_legend,annot=False)  #gist_rainbow    
        axs[2].set_title('$\hat{B}_{x^*}$')
        
        
            # sns.heatmap(Bhat_correct_estCov, ax=axs[1,1], cmap='PiYG', center=0, annot=True)
            # axs[1,1].set_title('$\hat{B}_{x^*c2}-B$')
        fig.tight_layout()
        plt.show()
        
        
        # ploting the 3D-scatter plot for all estimates for the averaged estimates
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(20,20), dpi=125)
        X = np.arange(0, dim_X, 1)
        Y = np.arange(0, dim_Y, 1)
        Y, X = np.meshgrid(Y, X)
        Z_trueB=B[Y,X]
        Z_B_lse=matrix_mean_B_hat_trueX[Y,X]
        Z_naive_Bhat=matrix_mean_B_hat_naive[Y,X]
        Z_inst_Bhat=matrix_mean_B_hat_inst[Y,X]
        Z4=np.zeros((dim_Y, dim_X))[Y,X]
        # Plot the surface.
        #ax.plot_surface(X, Y, Z_B, linewidth=0, antialiased=False, alpha=0.1, color="blue")
        #ax.plot_surface(X, Y, Z_Bxerr, linewidth=0, antialiased=False, alpha=0.5, color="red")
        ax.scatter(Y, X, Z_trueB, linewidth=0, antialiased=False, alpha=1, marker='o', color="blue", s=120, label="$B$")
        ax.scatter(Y, X, Z_B_lse, linewidth=0, antialiased=False, alpha=1, marker='s',color="yellow",  s=60, label="$\hat{B}$")
        ax.scatter(Y, X, Z_inst_Bhat, linewidth=0, antialiased=False, alpha=1, marker="^", color="red", s=50,  label="$\hat{B}_{V}$")
        ax.scatter(Y, X, Z_naive_Bhat, linewidth=0, antialiased=False, alpha=1, marker="X", color="green", s=90,  label="$\hat{B}_{x^*}$")
        
        #ax.scatter(Y, X, Z_correct_Bhat_estCov, linewidth=0, antialiased=False, alpha=0.8, marker='D', s=20, label="$\hat{B}_{x^*c2}$")
        ax.plot_surface(Y, X, Z4, linewidth=0, antialiased=False, alpha=0.5, color="grey")
        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=10)
        ax.set_xlabel('p')
        ax.set_ylabel('q')
        ax.set_zlabel(' ')
        ax.legend(loc=1)
        plt.show()
        
        ##################################
        ### avg Empirical standard errors for B hat
        ##################################
        
        #2. empirical cov of B hat's
 
        #calculate the empirical covariance matrix for the B hat under true X
        centered_vec_Bhat_trueX=vec_Bhat_array_trueX-np.array([vec_mean_B_hat_trueX,]*N_Simu).T
        empirical_cov_trueX=centered_vec_Bhat_trueX@centered_vec_Bhat_trueX.T/(N_Simu-1)
        Bhat_LSE_avgSEE=np.sqrt(empirical_cov_trueX.diagonal()).mean()
        print(f"The avgSEE (LSE): {Bhat_LSE_avgSEE}")
        
        
        #Let mean estimate as the center to calculate the empirical variance
        centered_vec_B_naive=vec_Bhat_array_naive-np.array([vec_mean_B_hat_naive,]*N_Simu).T
        empirical_cov_naive=centered_vec_B_naive@centered_vec_B_naive.T/(N_Simu-1)
        Bhat_naive_avgSEE=np.sqrt(empirical_cov_naive.diagonal()).mean()
        print(f"The avgSEE (naive): {Bhat_naive_avgSEE}")

      
        #Let mean estimate as the center to calculate the empirical variance
        centered_vec_B_inst=vec_Bhat_array_inst-np.array([vec_mean_B_hat_inst,]*N_Simu).T
        empirical_cov_inst=centered_vec_B_inst@centered_vec_B_inst.T/(N_Simu-1)
        print(f"diagonals of empirical_cov for IV estimator:{empirical_cov_inst.diagonal()}")
        Bhat_IV_avgSEE=np.sqrt(empirical_cov_inst.diagonal()).mean()
        print(f"The avgSEE (IV Estimate): {Bhat_IV_avgSEE}")

        ###############################
        # theoretical cov of estimators 
        ###############################
        
        #referenced term cov naive
        temp2=B@Sigma_X@LA.inv(Sigma_X+Sigma_Ex)
        temp2T=temp2.T
            
        # generate estimation data
        vec_vecT=np.zeros((dim_X**2, dim_X**2))
            
        for i in range(N_est_simu):
                
                _, X, X_star, V=data_generation(case=case,dim_Y=dim_Y, dim_X=dim_X, N_Obs=N_Obs,
                                                COV_X=None, COV_U=Sigma_U, COV_Ex=Sigma_Ex,COV_Ey=Sigma_Ey, 
                                                COV_R=Sigma_R,
                                                COV_V=Sigma_V, B=B,A=A)
                
                temp=X@X_star.T@LA.inv(X_star@X_star.T)
                tempT=temp.T
                vec_tempT=tempT.reshape(-1,1)
                vec_vecT+=vec_tempT@tempT.reshape(1,-1)                
                
        mean_vec_vecT=vec_vecT/N_est_simu

        #theoretical cov for B hat under true X
        theoretical_cov_trueX=np.kron(LA.inv(Sigma_X), Sigma_U)/N_Obs
        Bhat_LSE_avgSEM=np.sqrt(theoretical_cov_trueX.diagonal()).mean()
        print(f"The avgSEM for B hat (LSE): {Bhat_LSE_avgSEM}")
        
        # diff_cov_trueX=empirical_cov_trueX-theoretical_cov_trueX
        # Cov_LSE_avgBias=np.mean(diff_cov_trueX)
        # print(f"avgBias for diff cov (LSE):{Cov_LSE_avgBias}")
        # Cov_LSE_Frob=LA.norm(diff_cov_trueX)
        # print(f"L2-norm of diff_cov_trueX (LSE): {Cov_LSE_Frob}")
        # Cov_LSE_infinity_norm=abs(diff_cov_trueX).max()
        # print(f" The infinity norm of diff cov for B hat (LSE):{Cov_LSE_infinity_norm}")
                        
        #theoretical cov for naive estimate
        theoretical_cov_naive2=np.kron(LA.inv(Sigma_X+Sigma_Ex), Sigma_U)/N_Obs\
                                    +np.kron(np.eye(dim_X), B)@mean_vec_vecT@np.kron(np.eye(dim_X), B.T)\
                                    -temp2T.reshape(-1,1)@temp2T.reshape(1,-1)
        
        #print(f"diagonals in theoretical cov of B hat (naive):{theoretical_cov_naive2.diagonal()}")
        Bhat_naive_avgSEM=np.sqrt(theoretical_cov_naive2.diagonal()).mean()
        print(f"The avgSEM for B hat (naive): {Bhat_naive_avgSEM}")
        # diff_cov_naive_refe2=empirical_cov_naive-theoretical_cov_naive2
        # Cov_naive_avgBias=np.mean(diff_cov_naive_refe2)
        # print(f"avgBias for diff cov (naive):{Cov_naive_avgBias}")
        # Cov_naive_Frob=LA.norm(diff_cov_naive_refe2)
        # print(f"L2-norm of the diff_cov_naive_refe2 (naive): {Cov_naive_Frob}")
        # Cov_naive_infinity_norm=abs(diff_cov_naive_refe2).max()
        # print(f"The infinity norm of the diff cov (naive): {Cov_naive_infinity_norm}")
                      
            
        #theoretical cov for IV estimate
        S_XstarV=A@Sigma_V
        part1=np.kron(LA.inv(S_XstarV.T)@Sigma_V@LA.inv(S_XstarV),Sigma_U)/N_Obs
        
        # generate estimation data
        vec_vecT=np.zeros((dim_X**2, dim_X**2))
        
        for i in range(N_est_simu):
                
                _, X, X_star, V=data_generation(case=case,dim_Y=dim_Y, dim_X=dim_X, N_Obs=N_Obs,
                                                COV_X=Sigma_X, COV_U=Sigma_U, COV_Ex=Sigma_Ex,COV_Ey=Sigma_Ey, 
                                                COV_R=Sigma_R,
                                                COV_V=Sigma_V, B=B,A=A)
                
                temp=X@V.T@LA.inv(X_star@V.T)
                tempT=temp.T
                vec_tempT=tempT.reshape(-1,1)
                vec_vecT+=vec_tempT@tempT.reshape(1,-1)                
                
        mean_vec_vecT=vec_vecT/N_est_simu
        
        
        theoretical_cov_inst=part1+np.kron(np.eye(dim_X), B)@mean_vec_vecT@np.kron(np.eye(dim_X), B.T)\
                                            -B.T.reshape(-1,1)@B.T.reshape(1,-1)
            
        print(f"diagonals of the theoretical cov of B hat (iV estimate):{theoretical_cov_inst.diagonal()}")
        IV_diagonal=np.array(theoretical_cov_inst.diagonal())
        IV_diagonal[IV_diagonal<0]=0
        print(f"adjusted IV diagonal:{IV_diagonal}")
        Bhat_inst_avgSEM=np.sqrt(IV_diagonal).mean()
        print(f"The avgSEM for B hat (IV estimate):{Bhat_inst_avgSEM}")
            
        # diff_cov_correct=empirical_cov_correct-theoretical_cov_corrected
        # Cov_correct_avgBias=np.mean(diff_cov_correct)
        # print(f"avgBias for diff cov (correct with true cov):{Cov_correct_avgBias}")
        # Cov_correct_Frob=LA.norm(diff_cov_correct)
        # print(f"L2-norm of diff_cov (correct with true cov): {Cov_correct_Frob}")
        # Cov_correct_infinity_norm=abs(theoretical_cov_corrected).max()
        # print(f" Infinite norm of diff cov (correct with true cov): {Cov_correct_infinity_norm}")
            
        #4. calculate the CR% of 95% CI for each element of B
        vec_B=B.T.reshape(-1)
        vec_B_matrix=np.array([vec_B,]*N_Simu).T
        
        #LSE
        Var_B_LSE=np.diag(theoretical_cov_trueX)
        sd_B_LSE=np.array([np.sqrt(Var_B_LSE),]*N_Simu).T
        lower_bound_LSE=vec_Bhat_array_trueX+ss.norm.ppf(0.025)*sd_B_LSE
        upper_bound_LSE=vec_Bhat_array_trueX+ss.norm.ppf(0.975)*sd_B_LSE
        InCI_logical_LSE=np.greater(vec_B_matrix, lower_bound_LSE) & np.less(vec_B_matrix,upper_bound_LSE)
        InCI_numeric_LSE=InCI_logical_LSE*1
        In_LSE=np.mean(InCI_numeric_LSE, axis=1)*100
        Bhat_LSE_avgCR=np.mean(In_LSE)
        print(f"The avgCI 95% for Bhat (LSE):{Bhat_LSE_avgCR}")
        
        #naive case
        Var_B_naive=np.diag(theoretical_cov_naive2)
        sd_B_matrix_naive=np.array([np.sqrt(Var_B_naive),]*N_Simu).T
        lower_bound_naive=vec_Bhat_array_naive+ss.norm.ppf(0.025)*sd_B_matrix_naive
        upper_bound_naive=vec_Bhat_array_naive+ss.norm.ppf(0.975)*sd_B_matrix_naive
        InCI_Logical_naive=np.greater(vec_B_matrix,lower_bound_naive) & np.less(vec_B_matrix, upper_bound_naive)
        InCI_numeric_naive=InCI_Logical_naive*1
        InCI_naive=np.mean(InCI_numeric_naive,axis=1)*100
        #CR_naive=InCI_naive.reshape(dim_X, dim_Y).T
        Bhat_naive_avgCR=np.mean(InCI_naive)
        print(f"The avgCI 95% for Bhat (naive): {Bhat_naive_avgCR}")
        #IV estimate
        #Var_B_inst=np.diag(theoretical_cov_inst)
        sd_B_matrix_inst=np.array([np.sqrt(IV_diagonal),]*N_Simu).T
        lower_bound_inst=vec_Bhat_array_inst+ss.norm.ppf(0.025)*sd_B_matrix_inst
        upper_bound_inst=vec_Bhat_array_inst+ss.norm.ppf(0.975)*sd_B_matrix_inst
        InCI_Logical_inst=np.greater(vec_B_matrix, lower_bound_inst) & np.less(vec_B_matrix,upper_bound_inst)
        InCI_numeric_inst=InCI_Logical_inst*1
        InCI_inst=np.mean(InCI_numeric_inst, axis=1)*100
        #CR_correct=InCI_correct.reshape(dim_X, dim_Y).T
        Bhat_inst_avgCR=np.mean(InCI_inst)
        print(f"The avgCI 95% for Bhat (IV estimate):{Bhat_inst_avgCR}")

            
        end_time=time.time()
        print(f"time used: {(end_time-start_time)/60} mins")
        
        return np.array([avgBias_diff_B_trueX,Bhat_LSE_avgSEE,Bhat_LSE_avgSEM,Bhat_LSE_avgCR,
                         avgBias_diff_B_naive,Bhat_naive_avgSEE,Bhat_naive_avgSEM,Bhat_naive_avgCR,	
                         avgBias_diff_B_inst,Bhat_IV_avgSEE,Bhat_inst_avgSEM,Bhat_inst_avgCR	
                        # , avgBias_diff_B_correct_estCov,	Bhat_correct2_avgSEE,Bhat_correct2_Frob,Bhat_correct2_infinity_norm,
                        # Cov_LSE_avgBias,Cov_LSE_Frob,Cov_LSE_infinity_norm,	
                        # Cov_naive_avgBias,	Cov_naive_Frob,	Cov_naive_infinity_norm,	
                        # Cov_correct_avgBias,Cov_correct_Frob,Cov_correct_infinity_norm
                         ])
                         
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

# Gaussian Mixed Model Constructor (NOt Used Yet)
def gaussian_mixture_generator(n, p, n_comp, probs, means, covs, all_gaussin=True, dists=None):
                  """
                  Generate Multivariate Non-Normal distributed random variables.
                  Center at 0.
                  Args:
                      n (int): sample size of the mutivariate random data
                      p (int): the dimension of the multivariate random variable
                      n_comp (int): the number of gaussin components
                      probs (list): list of probabilities of all gaussian components, a vector of length n_comp
                      means (list): if p=1, means is a list of numbers of length n_comp, indicating the means of normal distributions
                                    if p>1, means is a list of vectors of length n_comp, indicating the mean vectors of Gaussian components
                      covs (list): if p=1, covs is a list of positive numbers of length n_comp;
                                    if p>1, covs is a list of matrices of dimensions p by p of length n_comp
                  """
                  ###Test Point 1###
                  #check the length of 
                  if len(probs)!=n_comp:
                                    raise SyntaxError('Length of probs is not equal to n_comp.')
                  if len(covs)!=n_comp:
                                    raise SyntaxError('Length of covs is not equal to n_comp.')
                  standard=[0]
                  for i in range(n_comp):
                                    standard=standard+[sum(probs[:i+1])]  
                  standard=standard
                  ###Test Point 2###
                  #print(f"standard:{standard}")
                                    
                  #when p=1, we generate univariate non-normal random variable
                  list_uniform=np.random.uniform(low=0, high=1, size=n)
                  #print(list_uniform)
                  
                  if p==1:                 
                                    rv=np.zeros(n)                  
                                    for i in range(n_comp):
                                                      #
                                                      ind=np.greater(list_uniform, standard[i]) & np.less(list_uniform, standard[i+1])   
                                                      #print(f"ind:{ind}")
                                                      N=sum(ind) 
                                                      #print(f"N:{N}") 
                                                      rv[ind]=np.random.normal(loc=means[i], scale=np.sqrt(covs[i]), size=N)
                                    return rv                       
                  elif p>1:
                                    rv=np.zeros((p,n))
                                    for i in range(n_comp):
                                                      ind=np.greater(list_uniform, standard[i]) & np.less(list_uniform, standard[i+1])   
                                                      #print(f"ind:{ind}")
                                                      N=sum(ind)
                                                      #print(f"N:{N}")   
                                                      rv[:,ind]=np.random.multivariate_normal(mean=means[i], cov=covs[i], size=N).T
                                    return rv
 
