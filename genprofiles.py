# %% Impor crease package
import crease_he
import numpy as np
import matplotlib.pyplot as plt

# %% Load shape
data = np.genfromtxt("IEXP_DATA/Itot_disper_20_Ain6_B12_Aout6_nLP7_dR0.2.txt")
q_range = data[:,0]
Iqexp = data[:,1]
iqs = []
q = []
sas = crease_he.InSilicoProfile(shape='vesicle',
                                 shape_params=[24,54,0.5,50.4,50.4,0.55,7])

# %% Generate profile
"""
params = [[200, 60, 120, 60, 0.2, 0.2, 3.51]]
#params = [[200, 60, 120, 60, 0.2, 0.2, 0.01+i*(5-0.1)/10] for i in range(0,10)]
for i in range(0,5):
        rho_B = 0.75+i*0.05
        sas = crease_he.InSilicoProfile(shape='vesicle',
                                 shape_params=[24,54,rho_B,50.4,50.4,0.55,7])
        print("\nrho_B=",rho_B)
        iq, q = sas.genprofile(params=params,
                                q_range = q_range,
                                #q_min= [0.003, 0.1, 50],
                                #output_dir='./ICOMP_DATA/',
                                #seed = 1,
                                plot = False)
        iqs.append(iq[0])
np.savetxt('./test_temp.txt',np.c_[iqs], fmt="%.8f")
"""
# %% Plot
iqs = np.genfromtxt("test_temp.txt")
colors = plt.cm.coolwarm(np.linspace(0,1,len(iqs)))
figsize=(4,4)
fig, ax = plt.subplots(figsize=(figsize))
for i in range(0,len(iqs)):
        ax.plot(q_range,iqs[i],color=colors[i],linestyle='-',ms=8,linewidth=1.3,marker=',',label=f'rho_B={0.25+i*0.05}')
ax.plot(q_range,Iqexp,color='k',linestyle='-',ms=8,linewidth=1.3,marker=',',label='Exp')
plt.xlim(q_range[0],q_range[-1])
plt.ylim(2*10**(-5),20)
plt.xlabel(r'q, $\AA^{-1}$',fontsize=20)
plt.ylabel(r'$I$(q)',fontsize=20)
ax.legend()
ax.set_xscale("log")
#ax.set_yscale("log")
plt.show()

# %%
