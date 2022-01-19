import numpy as np
import pandas as pd
from numpy import linalg as LA 

def cov_generator(N_VARIABLES, MODEL="exp", space_index=1):
    n_voxel=N_VARIABLES
    temp=int(np.sqrt(n_voxel)/2)
    a, b = np.meshgrid(np.arange(-temp, temp, 1), np.arange(-temp, temp, 1))
    a=np.reshape(a,-1)
    b=np.reshape(b,-1)
    dist=np.zeros((n_voxel, n_voxel))
    for i in range(n_voxel):
        for j in range(n_voxel):
            dist[i,j]=np.sqrt((a[i]-a[j])**2+(b[i]-b[j])**2)
    if MODEL=="exp":
        cov=np.exp(-space_index*dist)
    return cov

def matrix_generator(COV, N_OBSERVATIONS):
    """
        Return the data with dimension N_OBSERVATIONS*N_VARIABLES
        #(row of X)=N_OBSERVATIONS
        #(column of X)=COV.shape[0]
    """
    n_voxel=COV.shape[0]
    u,s,vh=LA.svd(COV)
    R=u@np.diag(np.sqrt(s))@vh
    X=np.random.normal(loc=0.0, scale=1.0, size=(n_voxel,N_OBSERVATIONS))
    X=R@X
    return X

def coeff_generator(COV_Y, COV_X, RANK):
    """
        RANK: is use to adjust the rank of B, so that full-rank and low-rank case can be both considered
        Return:
              B: the coefficient matrix with dimensions n_voxle_y by n_voxle_x
    """
    if RANK<=min(COV_X.shape[0], COV_Y.shape[0]):
        u_x,s_x,vh_x=LA.svd(COV_X)  
        u_y,s_y,vh_y=LA.svd(COV_Y)
        B_fullrank=u_y@np.diag(np.sqrt(s_y))@vh_y@np.random.normal(size=(COV_Y.shape[0],COV_X.shape[0]))@u_x@np.diag(np.sqrt(s_x))@vh_x
        #/np.sqrt(COV_X.shape[0]-1)
        u,s,vh=LA.svd(B_fullrank)
        s=s[:RANK]
        u=u[:,:RANK]
        vh=vh[:RANK,:]
        B=u@np.diag(s)@vh
    else:
        raise TypeError('Please make the rank smaller or equal to the smallest variables.')
    
    return B


def data_generator(N_VARIABLES_X, N_VARIABLES_Y, N_OBS, B=None, DATATYPE="accurate",
    MODEL={"X": "exp","Y": "exp"}, MODEL_INDEX={"X_IND":1, "Y_IND": 0.8}, RANK_COEFF=25, 
    N2S_RATIO_Z=0.5, N2S_RATIO_X=0.3, N2S_RATIO_Y=0.3):
    """
        Input: 
            DATATYPE: "accurate", "error-prone X", "error-prone Y", and "error-prone X Y"
        Return:
            X: the simulated covariates
            Y: the simulated response
            B: the true coefficient B
            COV_XX: the true covariance structure of the observed X
            COV_YY: the true covariance structure of the observed Y
            COV_YX: the true covariance structure of the observed Y and X
    """
    #Step 1 generate COV_X
    
    cov_x=cov_generator(N_VARIABLES=N_VARIABLES_X, MODEL=MODEL["X"], space_index=MODEL_INDEX["X_IND"])
    cov_y=cov_generator(N_VARIABLES=N_VARIABLES_Y, MODEL=MODEL["Y"], space_index=MODEL_INDEX["Y_IND"])
    #Step 2 generate X 
    X=matrix_generator(COV=cov_x, N_OBSERVATIONS=N_OBS)
    #Step 3 generate true coefficient matrix B
    if np.sum(B==None)==1:
        #B=coeff_generator(COV_Y=cov_y, COV_X=cov_x, RANK=RANK_COEFF)
        pass
    #Step 4 calculating the scale of error Z
    scale_Z=np.mean(np.sqrt(np.diag(B@cov_x@B.T)))*N2S_RATIO_Z
    #Step 5 generate error Z
    Z=np.random.normal(loc=0.0, scale=scale_Z, size=(N_VARIABLES_Y, N_OBS))
    COV_zz=np.eye(N_VARIABLES_Y)*(scale_Z**2)
    #Step 6 generate Y
    Y=B@X+Z
    COV_Y=B@cov_x@B.T+COV_zz
    COV_YX=B@cov_x

    if DATATYPE=="accurate":
        return X, Y, B, cov_x, COV_Y, COV_YX

    elif DATATYPE=="error-prone X":
        #Step 7 generate error-contaminated X*
        X_err=X+np.random.normal(loc=0.0, scale=N2S_RATIO_X, size=(N_VARIABLES_X,N_OBS))
        #calculating the COV_X of error-contaminated X
        cov_x_err=cov_x+np.eye(N_VARIABLES_X)*(N2S_RATIO_X**2)
        #Step 9 calculating the B*, the coefficient when the X is error-contaminated
        B_star=B@cov_x@LA.inv(cov_x_err)
        return X_err, Y, B_star, cov_x_err, COV_Y, COV_YX

    elif DATATYPE=="error-prone Y":
        #Step 8 generate error-prone Y*
        scale_Y=np.mean(np.sqrt(np.diag(COV_Y)))*N2S_RATIO_Y
        Y_err=Y+np.random.normal(loc=0.0, scale=scale_Y, size=(N_VARIABLES_Y, N_OBS))
        cov_y_err=COV_Y+np.eye(N_VARIABLES_Y)*(scale_Y**2)
        return X, Y_err, B, cov_x, cov_y_err, COV_YX

    elif DATATYPE=="error-prone X Y":
        #generate error-contaminated X*
        X_err=X+np.random.normal(loc=0.0, scale=N2S_RATIO_X, size=(N_VARIABLES_X, N_OBS))
        #calculating the COV_X of error-contaminated X
        cov_x_err=cov_x+np.eye(N_VARIABLES_X)*(N2S_RATIO_X**2)
        #Step 9 calculating the B*, the coefficient when the X is error-contaminated
        B_star=B@cov_x@LA.inv(cov_x_err)
        scale_Y=np.mean(np.sqrt(np.diag(COV_Y)))*N2S_RATIO_Y
        Y_err=Y+np.random.normal(loc=0.0, scale=scale_Y, size=(N_VARIABLES_Y, N_OBS))
        cov_y_err=COV_Y+np.eye(N_VARIABLES_Y)*(scale_Y**2)
        return X_err, Y_err, B_star, cov_x_err, cov_y_err, COV_YX
    else:
            raise TypeError('The datatype must be one of: "accurate", "error-prone X", "error-prone Y", and "error-prone X Y".')


def LS_fit(X, Y):
    n=X.shape[1]
    return (Y@X.T/(n-1))@LA.inv(X@X.T/(n-1))
    

def A_G(COV_YY, COV_XX, COV_YX):
    """
        Calculating A and G from the covariance matrices
    """
    #Step 1 calculating COV_YY^{-1/2}
    eigvalue_y, eigvector_y=LA.eig(COV_YY)
    inv_sqrt_Sigma_y=eigvector_y@np.diag(1/np.sqrt(eigvalue_y))@LA.inv(eigvector_y)
    
    #Step 2 calculating COV_XX^{-1/2}
    eigvalue_x, eigvector_x=LA.eig(COV_XX)
    inv_sqrt_Sigma_x=eigvector_x@np.diag(1/np.sqrt(eigvalue_x))@LA.inv(eigvector_x)

    #Step 3 SVD
    U, S, Vh=LA.svd(inv_sqrt_Sigma_y@COV_YX@inv_sqrt_Sigma_x)

    #A
    A=inv_sqrt_Sigma_y@U
    #G
    G=inv_sqrt_Sigma_x@Vh.T

    return A, G

def Lowrank_fit(X,Y, RANK):

    #sample size
    n=X.shape[1]
    #Calculating the sample cov
    Sigma_xx=X@X.T/n
    Sigma_yy=Y@Y.T/n
    Sigma_yx=Y@X.T/n
    #calculating A.hat and G.hat
    A_hat, G_hat=A_G(Sigma_yy,Sigma_xx,Sigma_yx)
    G1_hat=G_hat[:,:RANK]
    #A1_hat=A_hat[:,:RANK]

    #calculating the estimated low-rank coefficient matrix B
    B_k_hat=Sigma_yx@G1_hat@G1_hat.T

    return B_k_hat
  
  

def low_rank_simu(N_VARIABLES_X,N_VARIABLES_Y, B_fixed=None, N_OBS=1000, N_simu=30, DATATYPE="accurate",MODEL={"X": "exp","Y": "exp"}, 
    MODEL_INDEX={"X_IND":1, "Y_IND": 0.8}, RANK_COEFF=25, RANK_EST=25, N2S_RATIO_Z=0.5, N2S_RATIO_X=0.3, N2S_RATIO_Y=0.3):
    """
        Please remember to prespecify a coefficient matrix B
    """
    Diff_vectorized=np.zeros((N_VARIABLES_X*N_VARIABLES_Y,N_simu))
    
    for simu_id in range(N_simu):
        #
        #Step 1 generate random data and obtain the true covariance structure
        X_obs, Y_obs, B_converge, Sigma_XX, Sigma_YY, Sigma_YX=data_generator(N_VARIABLES_X, N_VARIABLES_Y, N_OBS=N_OBS, 
        B=B_fixed, DATATYPE=DATATYPE, MODEL=MODEL, MODEL_INDEX=MODEL_INDEX, RANK_COEFF=RANK_COEFF, 
        N2S_RATIO_Z=N2S_RATIO_Z, N2S_RATIO_X=N2S_RATIO_X, N2S_RATIO_Y=N2S_RATIO_Y)
        Sigma_zz=Sigma_YY-B_converge@Sigma_XX@B_converge.T

        #Step 2 calculating the estimated low-rank B based on the observed data X_obs and Y_obs
        B_k_hat=Lowrank_fit(X_obs,Y_obs, RANK=RANK_EST)

        #Step 4
        Diff_B=B_k_hat-B_converge
        #norm_diff_expect.append(np.sqrt(LA.norm(Diff_B)**2/(N_VARIABLES_X*N_VARIABLES_Y))/np.std(B_converge.reshape(-1)))
        Diff_vectorized[:,simu_id]=Diff_B.T.reshape(-1)
        #Step 8 mean norm of the difference between the B_k and B_converge
 
    #Step 6 calculating the empirical covariance
    emperical_cov=Diff_vectorized@Diff_vectorized.T*N_OBS/(N_simu-1)

    #Step 5 calculating the theoretical asymptotic covariance
    A, G=A_G(Sigma_YY, Sigma_XX, Sigma_YX)
    A1=A[:,:RANK_EST]
    G1=G[:,:RANK_EST]
    Lamb=Sigma_YX@G1
    P=G1
    part1=Sigma_zz-Lamb@LA.inv(Lamb.T@LA.inv(Sigma_zz)@Lamb)@Lamb.T
    part2=LA.inv(Sigma_XX)-P@LA.inv(P.T@Sigma_XX@P)@P.T
    COV_converge=np.kron(LA.inv(Sigma_XX),Sigma_zz)-np.kron(part2, part1)

    diff_cov=(emperical_cov-COV_converge).T.reshape(-1)/abs(COV_converge.T.reshape(-1))

    #calculate the mean of B difference between the B.hat and B.converge
    mean_diff_vec_B=np.mean(Diff_vectorized, axis=1)/abs(B_converge.T.reshape(-1))

    return mean_diff_vec_B, diff_cov
    #mean_diff_vec_B: vector with length of N_VARIABLES_X*N_VARIABLES_Y
    #diff_cov: vector with length of (N_VARIABLES_X*N_VARIABLES_Y)^2
