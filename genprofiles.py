# %% Impor crease package
import time
import crease_he
import numpy as np
import matplotlib.pyplot as plt

# %% Load shape
data = np.genfromtxt("IEXP_DATA/Itot_disper_20_Ain6_B12_Aout6_nLP7_dR0.2.txt")
q_range = data[:,0]
Iqexp = data[:,1]
iqs = []
q = []
#sas = crease_he.InSilicoProfile(shape='vesicle',
#                                 shape_params=[24,54,0.5,50.4,50.4,0.55,7])
def fitness(IQid):
        err=0
        qfin=q_range[-1]
        for qi,_ in enumerate(q_range):
            if (IQid[qi]>0)&(Iqexp[qi]>0):
                if (qi<qfin):
                    wil=np.log(np.true_divide(q_range[qi+1],q_range[qi]))  # weighting factor
                else:
                    wil=np.log(np.true_divide(q_range[qi],q_range[qi-1]))  # weighting factor
                err+=wil*(np.log(np.true_divide(Iqexp[qi],IQid[qi])))**2  # squared log error 
        #     elif metric == 'chi2':
        #         if IQerr is None:
        #             # chi^2 with weighting of IQin
        #             err += np.true_divide(
        #                 np.square(self.IQin[qi]-IQid[qi]), np.square(self.IQin[qi]))
        #         else:
        #             # chi^2 with weighting of IQerr
        #             err += np.true_divide(
        #                 np.square(self.IQin[qi]-IQid[qi]), np.square(self.IQerr[qi]))
        return err
# %% Generate profile
bg = 3.5
#10_12_6_12#3.0689#2.9656#
#20_6_12_6#3.3874#3.3976
params = [[100, 120, 120, 120, 0.2, 0.2, bg]]
sh_params = [54, 54, 0.5, 50.4, 50.4, 0.75, 7]
#params = [[200, 60, 120, 60, 0.2, 0.2, 0.01+i*(5-0.1)/10] for i in range(0,10)
#sh_params = [28, 46, 0.7198, 50.4, 41.1713, 0.5779, 7]
#20_6_12#[19, 67, 0.5368, 50.4, 59.9036, 0.5989, 7]#[15, 65, 0.6352, 50.4, 54.2969, 0.7840, 1]
#10_12_6_12#[28, 46, 0.7198, 50.4, 41.1713, 0.5779, 7]#[20, 45, 0.8789, 50.4, 38.125, 0.6047, 7]
#print("\n",sh_params)

for i in range(0,1):
    sas = crease_he.InSilicoProfile (shape='vesicle',
                                        shape_params=sh_params)#
    (iq, t) , q = sas.genprofile(params=params,
                            q_range = q_range,
                            #q_min= [0.003, 0.1, 50],
                            output_dir=['./ICOMP_DATA/',"8"],
                            #seed = 1,
                            plot = False)
    iqs.append(iq[0])
    #sse = fitness(iq[0])
    print("\nt:",t[0])#," sse:",sse)
np.savetxt('./test_temp.txt',np.c_[iqs], fmt="%.8f")

# %% Plot
iqs = np.genfromtxt("test_temp.txt")
if type(iqs[0]).__name__=='float64':
     iqs = np.array([iqs])
colors = plt.cm.coolwarm(np.linspace(0,1,len(iqs)))
figsize=(4,4)
fig, ax = plt.subplots(figsize=(figsize))
for i in range(0,len(iqs)):
    ax.plot(q_range,iqs[i],color=colors[i],linestyle='-',ms=8,linewidth=1.3,marker=',',label=f'ns={54 - i*3}')
#ax.plot(q_range,Iqexp,color='k',linestyle='-',ms=8,linewidth=1,marker=',',label='Exp')
plt.xlim(q_range[0],q_range[-1])
plt.ylim(2*10**(-5),20)
plt.xlabel(r'q, $\AA^{-1}$',fontsize=20)
plt.ylabel(r'$I$(q)',fontsize=20)
ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
plt.savefig('testplot.png',dpi=169,bbox_inches='tight')
plt.show()

# %%
