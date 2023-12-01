# %% Imports
from matplotlib import pyplot as plt
import numpy as np
import crease_he
import os
import datetime, time

if __name__ == "__main__":
    offTime = None
    #offTime = datetime.datetime(2023, 11, 28, 6, 0)
    remainOn = False
    fb = {"1_10_12_6_12_0.5_3.8" : 0.5,
          "2_10_6_12_6_0.7_4" : 0.7,
          "3_15_12_6_12_0.55_4.2" : 0.55,
          "4_15_6_12_6_0.8_4" : 0.8}
    iexps = ["1_10_12_6_12_0.5_3.8",
            "2_10_6_12_6_0.7_4",
            "3_15_12_6_12_0.55_4.2",
            "4_15_6_12_6_0.8_4"]
    #%% Upload data
    n = 50
    filenames = ["ga_s23_I2_10_6_12_6_0.7_4_n15_w7",
                 "ga_s22_I2_10_6_12_6_0.7_4_n15_w4",
                 "ga_s21_I2_10_6_12_6_0.7_4_n15_w1",
                 "ga_s20_I2_10_6_12_6_0.7_4_n15_w1"
                 ]
    for filename in filenames:
        print(filename)
        iexp = -1
        for Iexp in iexps:
            if Iexp in filename:
                iexp = Iexp
                break
        data = np.genfromtxt(f"ICOMP_DATA/{iexp}.txt")
        q_range = data[:,0]
        Iqexp = data[:,1]
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
            return err
        tempPath = f"./test_outputs/{filename}"
        pathFiles = f"./test_outputs/SSEvol_{filename}"
        if os.path.isfile(f"{pathFiles}/current_ID.txt"):
            sses = np.genfromtxt(f"{pathFiles}/sses.txt")
            iter = np.genfromtxt(f"{pathFiles}/iter.txt")
            individualID = np.genfromtxt(f"{pathFiles}/IDs.txt")
            uniqueIndividuals = np.genfromtxt(f"{pathFiles}/uniqueIndividuals.txt")
            currentID = int(np.genfromtxt(f"{pathFiles}/current_ID.txt"))
            bestSSE = np.genfromtxt(f"{pathFiles}/bestSSE.txt")
        else:
            os.mkdir(pathFiles)
            maxIter = int(np.genfromtxt(f'{tempPath}/current_cicle.txt'))
            iter = []
            bestparams = []
            bestSSE = []
            for i in range(maxIter):
                if os.path.isfile(f"{tempPath}/results_{i}.txt"):
                    tempdb = np.loadtxt(f"{tempPath}/results_{i}.txt")
                    parambesti = tempdb[np.argmin(tempdb[:,-1])]
                    bestSSE.append(parambesti[-1])
                    iter.append(i)
                    bestparams.append(parambesti[1:8])
            bestparams = np.array(bestparams)
            _, indices = np.unique(bestparams, axis=0, return_index=True)
            iter = np.array(iter)
            bestSSE = np.array(bestSSE)
            #individualIter = {}
            uniqueIndividuals = []
            individualID = np.ones((np.max(iter)+1,),dtype=int)*-1
            for i in range(len(indices)):
                index = indices[i]
                uniqueIndividual = bestparams[index]
                iters = iter[np.where(np.all(bestparams == bestparams[index], axis=1))[0]]
                individualID[iters] = i
                uniqueIndividuals.append(uniqueIndividual)
                #individualIter[i] = {"individual": uniqueIndividual,"iters": iters}
            np.savetxt(f"{pathFiles}/uniqueIndividuals.txt",np.c_[uniqueIndividuals])
            np.savetxt(f"{pathFiles}/IDs.txt",np.c_[individualID.T], fmt="%d")
            np.savetxt(f"{pathFiles}/iter.txt",np.c_[iter.T], fmt="%d")
            np.savetxt(f"{pathFiles}/bestSSE.txt",np.c_[bestSSE.T])
            currentID = 0
            np.savetxt(f"{pathFiles}/current_ID.txt",np.c_[currentID], fmt="%d")
            sses = np.ones((np.max(iter)+1,n))*10
        # %% Compute profile
        sh_params = [15, 30, 0.5, 50.4, 40, fb[iexp], 7]
        sas = crease_he.InSilicoProfile (shape = 'vesicle',
                                        shape_params = sh_params)#
        
        for i in range(currentID, len(indices)):
            print("ID:",i)
            params = [uniqueIndividuals[i] for k in range(n)]
            (iqs, t) , _ = sas.genprofile(params=params,
                                        q_range = q_range,
                                        #  seed = 1,
                                        n_cores = 8)
            sse = []
            np.savetxt(f"{pathFiles}/iqs_ID{i}.txt",np.c_[iqs])
            for iq in iqs:
                sse.append(fitness(iq))
            np.savetxt(f"{pathFiles}/SSEtime_ID{i}.txt",np.c_[sse, t])
            np.savetxt(f"{pathFiles}/current_ID.txt",np.c_[i+1], fmt="%d")
            sses[individualID==i] = np.array(sse)
            np.savetxt(f"{pathFiles}/sses.txt",np.c_[sses])
            #individualIter[i]["sse"] = sse
        #%% Boxplot
        fig1, ax1 = plt.subplots()
        ax1.plot(iter,bestSSE,color="r",linestyle='-',ms=8,linewidth=2, label="SSEreport")
        ax1.boxplot(sses.T, labels=iter)
        ax1.set_yscale("log")
        plt.xlabel(r'Generation',fontsize=20)
        plt.ylabel(r'bestSSE',fontsize=20)
        plt.savefig(f"{pathFiles}/SSEdisperEvol(2).png",dpi=169,bbox_inches='tight')
        time.sleep(200)

    # %% Shutt down
"""
    if offTime is not None:
        t_shut_down=10
        if remainOn:
            t_shutdown = offTime-datetime.datetime.now()
            t_shut_down = int(t_shutdown.total_seconds())
            if t_shut_down > 0:
                print("Shutting down time setted at",offTime)
                print("Current time ",datetime.datetime.now())
                print(f'Shutting down in {t_shutdown}')
            else:
                t_shut_down=10
        m.shut_down(t_shut_down)
        """