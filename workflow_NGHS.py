# %% Imports
import crease_he
import os
import datetime, time
import multiprocessing as mp
from functools import partial

# %% Work setup
#os.mkdir("./test_outputs")
algs = [
    "nghsmin",
    "nghsavgt",
    "nghsmint",
    "nghsavgt2"
    ]
alg = algs[0]
iexps = [
    "1_10_12_6_12",
    "2_10_6_12_6",
    "3_15_12_6_12",
    "4_15_6_12_6"
    ]
seeds = [
    0,3,6,9,12,15,
    17,18,21,24,27,30
    ]
hpis = [
    1,
    #6
    ]
TH = 3600 #total harmonies
PM = 0.14
HMS = 20
vars = ["hpi"] # to put in the file name
param_accuracy = [0, 0, 0, 0, 2, 2, 2]
n_cores = 6
t_rest = 300
offTime = None
# offTime = datetime.datetime(2023, 12, 18, 6, 0)
remainOn = True

# %% set works
works = {}
k = 0
for seed in seeds:
    for iexp in iexps:
        for hpi in hpis:
            works[k] = {"seed":seed, "iexp":iexp, "hpi":hpi}
            k+=1

firstwork = 0
w = range(firstwork,k)#[0,1,2,3,4,5]

# %%
min_vals = [50, 30, 30, 30, 0.1, 0.0, 2.0]
max_vals = [250, 200, 200, 200, 0.45, 0.45, 5.5]
fb = {
    "1_10_12_6_12" : 0.5,
    "2_10_6_12_6" : 0.7,
    "3_15_12_6_12" : 0.55,
    "4_15_6_12_6" : 0.8
    }

def crease(i, works, nc):
    iexp = works[i]["iexp"]
    seed = works[i]["seed"]
    hpi = works[i]["hpi"]
    print(f"WORK {i}: Iexp {iexp}, hpi:{hpi}, seed:{seed}\n")
    sha_params = [15, 30, 0.5, 50.4, 40, fb[iexp], 7]
    oparams = [HMS, int(TH/hpi), hpi, param_accuracy]#o_params[alg]
    aparams = [PM]#a_params[alg]#}#[0.85, 0.33, 0.01, 0.05, 0.01]
    m = crease_he.Model(optim_params = oparams,#[12, 5, 7],
                        adapt_params = aparams,#[0.005,0.85,0.1,1,0.006,0.25,1.1,0.6,0.001], 
                        opt_algorithm = alg,
                        work = i,
                        seed = seed)
                        #offTime = offTime)
    #detailed explanations are needed to describe what each value in shape params means
    m.load_shape(shape='vesicle',
                shape_params = sha_params,
                minvalu = min_vals,
                maxvalu = max_vals)
    #load target Iexp(q) IEXP_DATA  
    m.load_iq('./ICOMP_DATA/'+iexp+'.txt')
    if vars != []:
        name = ""
        for var in vars:
            value = locals()[var]
            name = name+var+str(value)+"_"
        name = name[:-1]
        name = "I"+iexp.split("_")[0]+"_"+name+"_s"+str(seed)+"_w"+str(i)
    else:
        name = "I"+iexp.split("_")[0]+"_s"+str(seed)+"_w"+str(i)
    m.solve(name = name,
            output_dir = './test_outputs',
            verbose = False,
            n_cores = nc)
    time.sleep(t_rest)

# Shutt down
def shut_down(t):
    print(f"SHUTTING DOWN in {t}s\n")
    os.system("shutdown /a")
    os.system("shutdown /s /f /t "+str(t))#h")

# %% Execute works
if __name__ == "__main__":
    if offTime is not None:
        t_shut_down = offTime-datetime.datetime.now()
        if t_shut_down.total_seconds()>0:
            print("Shutting down time setted at",offTime)
            print("Current time ",datetime.datetime.now())
            print("Shutting down in",t_shut_down)
            t_shut_down = int(t_shut_down.total_seconds())+10*60
            shut_down(t_shut_down)
        else:
            print("ERROR: SHUTTING DOWN TIME INVALID:",t_shut_down)

    t0 = datetime.datetime.now()
    #One work per core
    pool = mp.Pool(n_cores)
    partial_work = partial(crease,
                           works = works,
                           nc = 1)
    pool.map(partial_work,[i for i in w])
    pool.close()
    pool.join

    #One work at time
    # for i in w:
    #     crease(i, works, n_cores)
    
    print("Total seconds: "+str((datetime.datetime.now()-t0).total_seconds()))
    print("Current time: "+str(datetime.datetime.now()))
    
    #Shutt down
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
        shut_down(t_shut_down)