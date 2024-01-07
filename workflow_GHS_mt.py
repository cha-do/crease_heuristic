# %% Imports
import crease_he
import os
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

# %% Work setup
#os.mkdir("./test_outputs")
algs = [
    "ghsmt"
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
nts = [
    2,
    6
    ]
TH = 3600 #total harmonies
HMCR = 0.85
PAR = 0.33
HMS = 20
vars = ["nt"] # to put in the file name
param_accuracy = [0, 0, 0, 0, 2, 2, 2]
t_rest = 300
offTime = None
# offTime = datetime.datetime(2023, 12, 18, 6, 0)
remainOn = True

# %% set works
works = {}
k = 0
for seed in seeds:
    for iexp in iexps:
        for nt in nts:
            works[k] = {"seed":seed, "iexp":iexp, "nt":nt}
            k+=1

firstwork = 0
w = range(firstwork, k)#[0,1,2,3,4,5]

# %%
min_vals = [50, 30, 30, 30, 0.1, 0.0, 2.0]
max_vals = [250, 200, 200, 200, 0.45, 0.45, 5.5]
fb = {
    "1_10_12_6_12" : 0.5,
    "2_10_6_12_6" : 0.7,
    "3_15_12_6_12" : 0.55,
    "4_15_6_12_6" : 0.8
    }

def shut_down(t):
    print(f"SHUTTING DOWN in {t}s\n")
    os.system("shutdown /a")
    os.system("shutdown /s /f /t "+str(t))#h")

if __name__ == "__main__":
    for i in range(k):
        iexp = works[i]["iexp"]
        seed = works[i]["seed"]
        nt = works[i]["nt"]
        print(f"WORK {i}: Iexp {iexp}, nt:{nt}, seed:{seed}\n")
        sha_params = [15, 30, 0.5, 50.4, 40, fb[iexp], 7]
        oparams = [HMS, TH, nt, param_accuracy]#o_params[alg]
        aparams = [HMCR, PAR]#a_params[alg]#}#[0.85, 0.33, 0.01, 0.05, 0.01]
        m = crease_he.Modelmt(optim_params = oparams,#[12, 5, 7],
                            adapt_params = aparams,#[0.005,0.85,0.1,1,0.006,0.25,1.1,0.6,0.001], 
                            opt_algorithm = alg,
                            work = i,
                            seed = seed)
        #detailed explanations are needed to describe what each value in shape params means
        m.load_shape(shape='vesiclemt',
                    shape_params = sha_params,
                    minvalu = min_vals,
                    maxvalu = max_vals)
        #load target Iexp(q) IEXP_DATA  
        m.load_iq('./ICOMP_DATA/'+iexp+'.txt')
        #'''
        if vars != []:
            name = ""
            for var in vars:
                value = locals()[var]
                name = name+var+str(value)+"_"
            name = name[:-1]
            name = "I"+iexp.split("_")[0]+"_"+name+"_s"+str(seed)+"_w"+str(i)
        else:
            name = "I"+iexp.split("_")[0]+"_s"+str(seed)+"_w"+str(i)
        pop = m.first_improvisation(name = name,
                output_dir = './test_outputs',
                n_cores = 6,
                pop = 0
                )
        t0 = datetime.datetime.now()
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers = nt) as executor:
            futures = []
            for val in range(nt):
                future = executor.submit(m.solve,lock, val, 
                                        name = name,
                                        output_dir = './test_outputs',
                                        verbose = False,
                                        n_cores = 1)
                futures.append(future)
        print("Total seconds: "+str((datetime.datetime.now()-t0).total_seconds()))
        print("Current time: "+str(datetime.datetime.now()))

    # %% Shutt down
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