#%%
from calendar import c
import matplotlib.pyplot as plt
import numpy as np
import simu_fun_collection3 as simuFun
from numpy import linalg as LA
import scipy.stats as ss

# %%
B=np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=(2, 3))

Sigma_X=simuFun.cov_generator(dim=3, MODEL="random")

Sigma_U=simuFun.cov_generator(dim=2, MODEL="identity", identity_var=[1])

Sigma_Ex=simuFun.cov_generator(dim=3, MODEL="identity", identity_var=[1])

sum_B=0
vec_B_array=np.zeros((2*3,100))
sum_InCI=np.zeros(2*3)



def single_simulation_accurate(dim_Y, dim_X, N_Obs, COV_X, COV_U, COV_Ex, B):
    
    #Step 2. data generation

    Y_data,X_data=simuFun.data_generation(case="XY",dim_Y=dim_Y, dim_X=dim_X, N_Obs=N_Obs,
                            COV_X=COV_X, COV_U=COV_U, COV_Ex=COV_Ex, B=B)
    
    
    #Step 2. Estimating B

    B_hat=1/(N_Obs-1)*Y_data@X_data.T@LA.inv(X_data@X_data.T)

    #empirical Sigma_X
    COV_X_hat=X_data@X_data.T/(N_Obs-1)
    #empirical Sigma_U
    Residual=Y_data-B_hat@X_data
    COV_U_hat=Residual@Residual.T/(N_Obs-1)
    #empirical COV_B
    COV_B_hat=np.kron(LA.inv(COV_X_hat),COV_U_hat)

    #3. CR% for 95%CI
    vec_B=B.T.reshape(-1)
    vec_B_hat=B_hat.T.reshape(-1)
    Var_B=np.diag(COV_B_hat)
    lower_bound=vec_B_hat+ss.norm.ppf(0.025)*np.sqrt(Var_B)
    upper_bound=vec_B_hat+ss.norm.ppf(0.975)*np.sqrt(Var_B)
    InCI_Logical=np.greater(vec_B,lower_bound) & np.less(vec_B, upper_bound)
    InCI=InCI_Logical*1

    #Step 3. return the estimated B
    return B_hat, InCI

#%%
N_Obs=300
N_Simu=100
B_hat, InCI=single_simulation_accurate(case="XY",dim_Y=2, dim_X=3, N_Obs=N_Obs, 
                            COV_X=Sigma_X, COV_U=Sigma_U, COV_Ex=Sigma_Ex, B=B)

B_hat,B, 
#%%
u,s,vh=LA.svd(Sigma_X)
R=u@np.diag(np.sqrt(s))@vh
Z=np.random.normal(loc=0.0, scale=1.0, size=(3,3000))
X=R@Z

#%%
eig_val, eig_vec=LA.eig(Sigma_X)
R2=eig_vec@np.diag(np.sqrt(eig_val))@eig_vec.T
#%%
N_Obs=1000
Y_data,X_data=simuFun.data_generation(case="XY",dim_Y=2, dim_X=3, N_Obs=N_Obs,
                            COV_X=Sigma_X, COV_U=Sigma_U, COV_Ex=Sigma_Ex, B=B)
Y_bar=np.mean(Y_data, axis=1)

#%%
Y_bar=np.mean(Y_data, axis=1)
Y_bar_matrix=np.array([Y_bar,]*N_Obs).T
Y_center=Y_data-Y_bar_matrix
X_bar=np.mean(X_data, axis=1)
X_bar_matrix=np.array([X_bar,]*N_Obs).T
X_center=X_data-X_bar_matrix
S_YX=Y_center@X_center.T/(N_Obs-1)
S_XX=X_center@X_center.T/(N_Obs-1)
B_hat=S_YX@LA.inv(S_XX)
B_hat, B
#%%
a=np.eye(3)
b=np.eye(3)
c=np.outer(a.reshape(-1),a.reshape(-1))
#plt.imshow(c)
a=np.random.normal(0,1,size=(4,5))
a.min()
#%%
A=simuFun.cov_generator(dim=3, MODEL="random")
print(A)
L=LA.cholesky(A)
L
#%%
a=[1,2,3,4,5,6,7,8,9]
b=np.array(a)
b.reshape(3,3)

#%%
dim_X=3
Big_0=[0]*(dim_X**2+(dim_X-1))+[1]
small_0=[0]*dim_X+[1]
Sets=Big_0*(dim_X-1)+small_0
M=[1]+Sets*(dim_X-1)+Big_0*(dim_X-1)
M=np.array(M)
M_q=M.reshape(9,9)