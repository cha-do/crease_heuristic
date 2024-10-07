#%%
import numpy as np
import matplotlib.pyplot as plt

k = 6000
c = 0.85
# plt.rcParams['font.family'] = 'Times New Roman'
# Iexps = [
#     # "1_10_12_6_12",
#     # "2_10_6_12_6",
#     "3_15_12_6_12",
#     # "4_15_6_12_6",
#     ]
Iexp = "3_15_12_6_12"
algs = [
    "GA",
    "NGHS"
]
colors = {"GA" : "r",
          "NGHS" : "g"}
fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(8*0.5, 8*0.5))#, sharex=True)#, sharey=True)
B=2

data = np.genfromtxt(f"../ICOMP_DATA/{Iexp}.txt")#ICOMP_DATA/7_15_6_12_6.txt")
q = data[:,0]
I = data[:,1]
# ax3.plot(numStrucs[mask],ParamBest["bestSSE"][mask],color=colors[o%len(colors)],linestyle='-',ms=8,linewidth=1,marker='')#,label="CREASE-NGHS")  # 'auto' lets Matplotlib decide the number of bins
ax3.plot(q,I,color='k',linestyle='-',ms=8,linewidth=3,marker='.',label=r'${I}_{exp}(Q)$')
# ax3.set_xlim(q[0],q[-1])
# ax3.set_ylim(2*10**(-5),20)
# fig3.text(0.08, 0.5, r'$I$(Q)', va='center', rotation='vertical', fontsize=10)
# fig3.text(0.5, 0.04, r'Q, [$\AA^{-1}$]', ha='center', fontsize=10)
# plt.xlabel(r'Q, $\AA^{-1}$',fontsize=20)
# plt.ylabel(r'$I$(Q)',fontsize=20)
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_ylabel(r'$I$(Q)',fontsize=10)
ax3.set_xlabel(r'Q, [$\AA^{-1}$]',fontsize=10)
# plt.savefig('testplot.png',dpi=169,bbox_inches='tight')
# ax3.grid()
# ax3.text(0.2, 0.2, f'B{B+1}', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes, fontsize=10)
ax3.tick_params(axis='both', labelsize=10)
if algs != []:
    for alg in algs:
        data = np.genfromtxt(f"/best_exc/{alg}/{Iexp}.txt")
        I = data[-1]
        ax3.plot(q,I,color=colors[alg],linestyle='-',ms=0,linewidth=2,marker='.',label=f"CREASE-{alg}")

ax3.legend(fontsize=8)
fig3.subplots_adjust(left=0.16)
fig3.savefig(f"SASprofilesBench_bestGA_NGHS_1Bt.tiff",format="tiff",dpi=600,bbox_inches='tight')
fig3.savefig(f"SASprofilesBench_bestGA_NGHS_1B.png",format="png",dpi=600,bbox_inches='tight')
    
