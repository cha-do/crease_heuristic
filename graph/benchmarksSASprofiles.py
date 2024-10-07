#%%
import numpy as np
import matplotlib.pyplot as plt

k = 6000
c = 0.85
plt.rcParams['font.family'] = 'Times New Roman'
Iexps = [
    "1_10_12_6_12",
    "2_10_6_12_6",
    "3_15_12_6_12",
    "4_15_6_12_6",
    ]
algs = [
    "GA",
    "NGHS"
]
colors = {"GA" : "r",
          "NGHS" : "g"}
fig3, ax3 = plt.subplots(nrows=2, ncols=2, figsize=(8*0.75, 8*0.75))#, sharex=True)#, sharey=True)
B=0
for Iexp in Iexps:
    data = np.genfromtxt(f"../ICOMP_DATA/{Iexp}.txt")#ICOMP_DATA/7_15_6_12_6.txt")
    q = data[:,0]
    I = data[:,1]
    yy = B%2
    xx = int(B>1)
    # ax3[xx,yy].plot(numStrucs[mask],ParamBest["bestSSE"][mask],color=colors[o%len(colors)],linestyle='-',ms=8,linewidth=1,marker='')#,label="CREASE-NGHS")  # 'auto' lets Matplotlib decide the number of bins
    ax3[xx,yy].plot(q,I,color='k',linestyle='-',ms=3.5,linewidth=1.5,marker='.',label=r'${I}_{exp}(Q)$')
    # ax3[xx,yy].set_xlim(q[0],q[-1])
    # ax3[xx,yy].set_ylim(2*10**(-5),20)
    fig3.text(0.08, 0.5, r'$I$(Q)', va='center', rotation='vertical', fontsize=10)
    fig3.text(0.5, 0.04, r'Q, [$\AA^{-1}$]', ha='center', fontsize=10)
    # plt.xlabel(r'Q, $\AA^{-1}$',fontsize=20)
    # plt.ylabel(r'$I$(Q)',fontsize=20)
    ax3[xx,yy].set_xscale("log")
    ax3[xx,yy].set_yscale("log")
    # plt.savefig('testplot.png',dpi=169,bbox_inches='tight')
    # ax3[xx,yy].grid()
    ax3[xx,yy].text(0.2, 0.2, f'B{B+1}', horizontalalignment='center', verticalalignment='center', transform=ax3[xx,yy].transAxes, fontsize=10)
    ax3[xx,yy].tick_params(axis='both', labelsize=10)
    if algs != []:
        for alg in algs:
            data = np.genfromtxt(f"best_exc/{alg}/{Iexp}.txt")
            I = data[-1]
            ax3[xx,yy].plot(q,I,color=colors[alg],linestyle='-',ms=0,linewidth=1,marker='.',label=f"CREASE-{alg}")
    B+=1
if algs != []:
    B = 1
    yy = B%2
    xx = int(B>1)
    ax3[xx,yy].legend(fontsize=8)
fig3.subplots_adjust(left=0.16)
fig3.savefig(f"SASprofilesBench_bestGA_NGHSt.tiff",format="tiff",dpi=600,bbox_inches='tight')
fig3.savefig(f"SASprofilesBench_bestGA_NGHS.png",format="png",dpi=600,bbox_inches='tight')
    
