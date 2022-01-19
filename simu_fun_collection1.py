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

def matrix_generator(N_OBSERVATIONS, COV):
    """
        Return the data with dimension N_OBSERVATIONS*N_VARIABLES
        #(row of X)=N_OBSERVATIONS
        #(column of X)=COV.shape[0]
    """
    n_voxel=COV.shape[0]
    u,s,vh=LA.svd(COV)
    R=u@np.diag(np.sqrt(s))@vh
    X=np.random.normal(loc=0.0, scale=1.0, size=(N_OBSERVATIONS,n_voxel))
    X=X@R.T
    return X

def coeff_generator(COV_X, COV_Y,RANK):
    """
        RANK: is use to adjust the rank of B, so that full-rank and low-rank case can be both considered
    """
    if RANK<=min(COV_X.shape[0], COV_Y.shape[0]):
        u_x,s_x,vh_x=LA.svd(COV_X)
        u_y,s_y,vh_y=LA.svd(COV_Y)
        B_fullrank=u_x@np.diag(np.sqrt(s_x))@vh_x@np.random.normal(size=(COV_X.shape[0], COV_Y.shape[0]))@u_y@np.diag(np.sqrt(s_y))@vh_y/np.sqrt(COV_X.shape[0]-1)
        u,s,vh=LA.svd(B_fullrank)
        s=s[:RANK]
        u=u[:,:RANK]
        vh=vh[:RANK,:]
        B=u@np.diag(s)@vh
    else:
        raise TypeError('Please make the rank smaller or equal to the smallest variables.')
    
    return B

def data_generator(N_VARIABLES_X, N_VARIABLES_Y, N_OBS, 
    MODEL={"X": "exp","Y": "exp"}, MODEL_INDEX={"X_IND":1, "Y_IND": 0.8}, RANK_COEFF=25, 
    N2S_RATIO_Z=0.5, N2S_RATIO_X=0.3, N2S_RATIO_Y=0.3):
    """
        Return:
            X: the simulated covariates
            Y: the simulated response
            B: the true coefficient B
    """
    #Step 1 generate COV_X
    cov_x=cov_generator(N_VARIABLES=N_VARIABLES_X, MODEL=MODEL["X"], space_index=MODEL_INDEX["X_IND"])
    cov_y=cov_generator(N_VARIABLES=N_VARIABLES_Y, MODEL=MODEL["Y"], space_index=MODEL_INDEX["Y_IND"])
    #Step 2 generate X 
    X=matrix_generator(N_OBSERVATIONS=N_OBS, COV=cov_x)
    #Step 3 generate true coefficient matrix B
    B=coeff_generator(COV_X=cov_x, COV_Y=cov_y,RANK=RANK_COEFF)
    #Step 4 calculating the scale of error Z
    scale_Z=np.mean(np.sqrt(np.diag(B.T@cov_x@B)))*N2S_RATIO_Z
    #Step 5 generate error Z
    Z=np.random.normal(loc=0.0, scale=scale_Z, size=(N_OBS,N_VARIABLES_Y))
    COV_zz=np.eye(N_VARIABLES_Y)*(scale_Z**2)
    #Step 6 generate Y
    Y=X@B+Z

    #Step 7 generate error-contaminated X*
    X_err=X+np.random.normal(loc=0.0, scale=N2S_RATIO_X, size=(N_OBS,N_VARIABLES_X))
    #calculating the COV_X of error-contaminated X
    cov_x_err=cov_x+np.eye(N_VARIABLES_X)*(N2S_RATIO_X**2)

    #Step 8 generate error-prone Y*
    COV_Y=B.T@cov_x@B+COV_zz
    scale_Y=np.mean(np.sqrt(np.diag(COV_Y)))*N2S_RATIO_Y
    Y_err=Y+np.random.normal(loc=0.0, scale=scale_Y, size=(N_OBS,N_VARIABLES_Y))
    cov_y_err=COV_Y+np.eye(N_VARIABLES_Y)*(scale_Y**2)

    #Step 9 calculating the B*, the coefficient when the X is error-contaminated
    B_star=LA.inv(cov_x+np.eye(N_VARIABLES_X)*N2S_RATIO_X**2)@cov_x@B

    return X, Y, B, X_err, Y_err, B_star, cov_x, COV_Y, cov_x_err, cov_y_err, COV_zz

def LS_fit(X, Y):
    n=X.shape[0]
    return LA.inv(X.T@X/(n-1))@X.T@Y/(n-1)

def low_rank_simu(COV_X, COV_Y, B, RANK, COV_zz, N2S_RATIO_Z=0.5, ):

    N_Y=COV_Y.shape[0]
    N_X=COV_X.shape[0]
    N_OBS=X.shape[0]

    #calculate Sigma_xx^{1/2}
    eigvalue_covx, eigvector_covx=LA.eig(COV_X)
    #sqrt_Sigma_X=eigvector_covx@np.diag(np.sqrt(eigvalue_covx))@LA.inv(eigvector_covx)
    inv_sqrt_Sigma_X=eigvector_covx@np.diag(1/np.sqrt(eigvalue_covx))@LA.inv(eigvector_covx)
    #calculate Sigma_yy^{1/2}
    eigvalue_covy, eigvector_covy=LA.eig(COV_Y)
    #sqrt_Sigma_Y=eigvector_covy@np.diag(np.sqrt(eigvalue_covy))@LA.inv(eigvector_covy)
    inv_sqrt_Sigma_Y=eigvector_covy@np.diag(1/np.sqrt(eigvalue_covy))@LA.inv(eigvector_covy)
    #calculating the Sigma_yx
    COV_YX=B@COV_X

    #calculating U
    Temp=inv_sqrt_Sigma_Y@COV_YX@inv_sqrt_Sigma_X
    U,s,Vh=LA.svd(Temp)
    R_bar=np.concatenate((np.diag(s),np.zeros((N_Y,N_X-N_Y))), axis=1)

    #The theoretical A and G
    A=inv_sqrt_Sigma_Y@U
    G=inv_sqrt_Sigma_X@Vh.T

    A1=A[:,:RANK]
    G1=G[:,:RANK]

    #calculating hat{B_k}
    B_k_hat=G1@G1.T@X.T@Y/(N_OBS-1)

    #calculating the theoretical asymptotic covariance

    #calculating the empirical covariance

    return B_k_hat
    #, B_converge, COV_empirical, COV_theoretical
