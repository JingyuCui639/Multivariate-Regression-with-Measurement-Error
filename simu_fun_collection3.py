import numpy as np
from numpy import linalg as LA 
import math

def cov_generator(dim, MODEL="random", random_sd=1, space_index=1,identity_sd=[1]):
    """
    dim: an integer indicating the dimension of the covariance matrix.
    MODEL: a string indicating the model to generate the covariance: "random",
            "identity", "exp",...
    random_sd: for the "random" model, this sets the sd when first generate 
              N(0,1) rvs.
    space_index: the parameter use when MODEL is "exp".
    identity_var: the var on the diagonal of the final matrix, should be
                 a list or an array of diagnal.    
    """
    identity_sd=np.array(identity_sd)
    
    if MODEL=="random":
        Z=np.random.normal(0,random_sd,(dim,dim))
        W=0.5*(Z+Z.T)
        eig_val, _=LA.eig(W)
        Sigma=W+np.eye(dim)*math.ceil(abs(eig_val.min()))
        return Sigma
    elif MODEL=="identity":
        if len(identity_sd)==dim:
            Sigma=np.eye(dim)@np.diag(identity_sd**2)
            return Sigma
        elif len(identity_sd)==1:
            Sigma=np.eye(dim)*(identity_sd**2)
            return Sigma

def data_generation(case,dim_Y, dim_X, N_Obs,COV_X, COV_U, COV_Ex, COV_Ey, COV_R, 
                            COV_V, B, A):
    """
        case: string: "XY", "X*Y", "XY*", "X*Y*"
        dim_Y: dimension of response vector (Y_i)
        dim_X: dimension of covariates vector (X_i)
        N_Obs: the number of observations
        COV_X: the covariance matrix of the covariates (X_i)
        COV_U: the covariance matrix of the error (U_i)
        COV_EX: the covariance matrix of the measurement error (E_xi)
    """
    #2. generate U from COV_U

    u,s,vh=LA.svd(COV_U)
    R=u@np.diag(np.sqrt(s))@vh
    Z=np.random.normal(loc=0.0, scale=1.0, size=(dim_Y,N_Obs))
    U=R@Z

    if case=="X*YV":
              u,s,vh=LA.svd(COV_V)
              R=u@np.diag(np.sqrt(s))@vh
              Z=np.random.normal(loc=0.0, scale=1.0, size=(dim_X,N_Obs))
              V=R@Z 

              u,s,vh=LA.svd(COV_R)
              R=u@np.diag(np.sqrt(s))@vh
              Z=np.random.normal(loc=0.0, scale=1.0, size=(dim_X,N_Obs))
              R_err=R@Z

              X=A@V+R_err

              Y=B@X+U

              u,s,vh=LA.svd(COV_Ex)
              R=u@np.diag(np.sqrt(s))@vh
              Z=np.random.normal(loc=0.0, scale=1.0, size=(dim_X,N_Obs))
              Ex=R@Z
              X_star=X+Ex

              return Y, X_star, V

    #1. generate X_{q\times n} from COV_X

    u,s,vh=LA.svd(COV_X)
    R=u@np.diag(np.sqrt(s))@vh
    Z=np.random.normal(loc=0.0, scale=1.0, size=(dim_X,N_Obs))
    X=R@Z

    #3. construct Y
    Y=B@X+U

    #If the measurement error in X case
    if case=="XY":
        return Y, X
    elif case=="X*Y":
        u,s,vh=LA.svd(COV_Ex)
        R=u@np.diag(np.sqrt(s))@vh
        Z=np.random.normal(loc=0.0, scale=1.0, size=(dim_X,N_Obs))
        Ex=R@Z
        X_star=X+Ex
        return Y, X_star
    elif case=="XY*":
        u,s,vh=LA.svd(COV_Ey)
        R=u@np.diag(np.sqrt(s))@vh
        Z=np.random.normal(loc=0.0, scale=1.0, size=(dim_Y,N_Obs))
        Ey=R@Z
        Y_star=Y+Ey
        return Y_star, X
    elif case=="X*Y*":
        u,s,vh=LA.svd(COV_Ex)
        R=u@np.diag(np.sqrt(s))@vh
        Z=np.random.normal(loc=0.0, scale=1.0, size=(dim_X,N_Obs))
        Ex=R@Z
        X_star=X+Ex

        u,s,vh=LA.svd(COV_Ey)
        R=u@np.diag(np.sqrt(s))@vh
        Z=np.random.normal(loc=0.0, scale=1.0, size=(dim_Y,N_Obs))
        Ey=R@Z
        Y_star=Y+Ey
        return Y_star, X_star

        

        
