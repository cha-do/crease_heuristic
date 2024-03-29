# %% Impor crease package
import time
import crease_he
import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":
    # %% Load shape
    data = np.genfromtxt("ICOMP_DATA/1_10_12_6_12_0.5.txt")
    q_range = data[:,0]
    Iqexp = data[:,1]
    # dq = 0.0011#0.0019903845283018866
    # qmin = 0.003
    # #qmax = 0.1
    # n = 53#int((qmax-qmin)/dq)+1
    # q_range = np.array(range(n))*dq+qmin
    # q_range = np.round(q_range, 8)
    iqs = []
    qs = []
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
    bg = 3.8
    #10_12_6_12#3.0689#2.9656#
    #20_6_12_6#3.3874#3.3976
    params = [[100, 120, 60, 120, 0.2, 0.2, bg]]
    sh_params = [30, 30, 0.5, 50.4, 40, 0.5, 7]
    n = 1
    #params = [[200, 60, 120, 60, 0.2, 0.2, 0.01+i*(5-0.1)/10] for i in range(0,10)
    #sh_params = [28, 46, 0.7198, 50.4, 41.1713, 0.5779, 7]
    #20_6_12#[19, 67, 0.5368, 50.4, 59.9036, 0.5989, 7]#[15, 65, 0.6352, 50.4, 54.2969, 0.7840, 1]
    #10_12_6_12#[28, 46, 0.7198, 50.4, 41.1713, 0.5779, 7]#[20, 45, 0.8789, 50.4, 38.125, 0.6047, 7]
    #print("\n",sh_params)
    T = []
    #dq = [0.0008,0.0012,0.0016,0.002]
    #nq = []
    #qrange = np.linspace(0.003, 0.18, num = 100)
    for i in range(1):
        #q_range = qrange[10*i:]
        sas = crease_he.InSilicoProfile (shape = 'vesicle',
                                         shape_params = sh_params)#
        #nq.append(int((0.1-0.03)/dq[i]))
        #dq[i] = round((0.1-0.03)/nq[i],5)
        (iq, t) , q = sas.genprofile(params=params,
                                     q_range = q_range,
                                    #q_lim = [0.003, 0.1, nq[i]],
                                    output_dir = ['./ICOMP_DATA/',f"{n}"],
                                    #seed = 1,
                                     plot = False)
        iqs.append(iq[0])
        qs.append(q)
        T.append(round(t[0],2))
        #sse = fitness(iq[0])
        print("\tt:", T[i])#," sse:",sse)
    # np.savetxt('./test_temp.txt',np.c_[np.array(iqs)], fmt="%.8f")
    # np.savetxt('./test_tempq.txt',np.c_[np.array(qs)], fmt="%.8f")
    # %% Plot
    # iqs = np.genfromtxt("test_temp.txt")
    # qs = np.genfromtxt("test_tempq.txt")
    if type(iqs[0]).__name__ == 'float64':
        iqs = np.array([iqs])
        qs = np.array([qs])
    colors = plt.cm.coolwarm(np.linspace(0,1,len(iqs)))
    figsize = (4,4)
    fig, ax = plt.subplots(figsize=(figsize))
    ax.plot(q_range,Iqexp,color='k',linestyle='-',ms=8,linewidth=1.2,marker=',',label='inf')
    for i in range(0,len(iqs)):
        ax.plot(q_range,iqs[i],color=colors[i],linestyle='-',ms=8,linewidth=1.3,marker=',',label=f'{bg}')
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
