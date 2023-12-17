# %% Imports
import crease_he
import os
import datetime, time
import multiprocessing as mp
from functools import partial

# %% Work setup
#os.mkdir("./test_outputs")
param_accuracy = [2, 2, 2, 2, 2, 2, 2]
o_params = {"ga" : [80, 100, 7],
            "ghs" : [20, 500, 6, param_accuracy],#HMS, TotalIter, newHarm/Iter
            "sghs" : [20, 700, 6]}#HMS, TotalIter, newHarm/Iter
a_params = {"ga" : [0.005,0.85,0.1,1,0.006,0.25,1.1,0.6,0.001],
            "ghs" : [0.85, 0.33],
            "sghs" : [0.85, 0.33, 0.01, 0.05, 0.01]}
alg = "ghs"
iexps = [
    "1_10_12_6_12",
    "2_10_6_12_6",
    "3_15_12_6_12",
    "4_15_6_12_6"
    ]
n_cores = 6
t_rest = 300
seeds = [23, 22]
offTime = None
# offTime = datetime.datetime(2023, 12, 18, 6, 0)
remainOn = True
firstwork = 0

# %% 
works = {}
k = 0
for seed in seeds:
    for iexp in iexps:
        works[k] = {"seed":seed, "iexp":iexp}
        k+=1

min_vals = [50, 30, 30, 30, 0.1, 0.0, 2.5]
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
    print(f"WORK {i}: Iexp {iexp}, seed:{seed}\n")
    sha_params = [15, 30, 0.5, 50.4, 40, fb[iexp], 7]
    oparams = o_params[alg]
    aparams = a_params[alg]#[0.85, 0.33]#[0.85, 0.33, 0.01, 0.05, 0.01]
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
    m.solve(name = "s"+str(seed)+"_I"+iexp+"_w"+str(i),
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
                           works=works,
                           nc = 1)
    pool.map(partial_work,[i for i in range(firstwork, k)])
    pool.close()
    pool.join

    #One work at time
    # for i in range(firstwork, k):
    #     crease(i, works, 1)
    
    print("Total seconds: "+str((datetime.datetime.now()-t0).total_seconds()))
    
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