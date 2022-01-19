#%matplotlib notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA #this package is used when calculating the norm
import simu_fun_collection1 as fun_old
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

X, Y, B, X_err, Y_err, B_star, cov_x, COV_Y, cov_x_err, cov_y_err, COV_zz=fun_old.data_generator(N_VARIABLES_X=64, N_VARIABLES_Y=36, N_OBS=10000,
MODEL={"X": "exp","Y": "exp"}, MODEL_INDEX={"X_IND":1, "Y_IND": 0.8}, RANK_COEFF=25, N2S_RATIO_Z=0.5, N2S_RATIO_X=0.3, N2S_RATIO_Y=0.3)

#B_xerr=fun_old.LS_fit(X_err, Y)
#B_xy=fun_old.LS_fit(X, Y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(11,9))
X = np.arange(0, 64, 1)
Y = np.arange(0, 36, 1)
X, Y = np.meshgrid(X, Y)
Z_Bstar=B_star[X,Y]
Z_B=B[X,Y]
Z3=Z_B-Z_Bstar
Z4=np.zeros((64,36))[X,Y]
# Plot the surface.
#ax.plot_surface(X, Y, Z_B, linewidth=0, antialiased=False, alpha=0.1, color="blue")
#ax.plot_surface(X, Y, Z_Bxerr, linewidth=0, antialiased=False, alpha=0.5, color="red")
ax.scatter(X, Y, Z_B, linewidth=0, antialiased=False, alpha=0.5, marker='o', label="$B$")
ax.scatter(X, Y, Z_Bstar, linewidth=0, antialiased=False, alpha=0.5, marker='^', label="$B^*=B\Sigma_{XX}(\Sigma_{XX}+\sigma_{\mathcal{E}^x}^2)^{-1}$")
#ax.scatter(X, Y, Z3, linewidth=0, antialiased=False, alpha=0.1, color="red")
ax.plot_surface(X, Y, Z4, linewidth=0, antialiased=False, alpha=0.5, color="grey")
# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend(loc=9)

plt.show()

