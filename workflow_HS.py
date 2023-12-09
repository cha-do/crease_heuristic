# %% Imports
import crease_he
import os
import datetime, time

if __name__ == "__main__":
    offTime = None
    offTime = datetime.datetime(2023, 12, 11, 6, 0)
    remainOn = True
    #os.mkdir("./test_outputs")
    fb = {
          "1_10_12_6_12_0.5_3.8" : 0.5,
          "2_10_6_12_6_0.7_4" : 0.7,
          "3_15_12_6_12_0.55_4.2" : 0.55,
          "4_15_6_12_6_0.8_4" : 0.8
          }
    # Iexps = {"10_12_6_12" : "10_Ain12_B6_Aout12_nLP7_dR0.2", #
    #         "10_12_12_12" : "10_Ain12_B12_Aout12_nLP7_dR0.2", #
    #         "20_6_12_6" : "20_Ain6_B12_Aout6_nLP7_dR0.2", #
    #         "30_12_6_12" : "30_Ain12_B6_Aout12_nLP7_dR0.2"}
    param_accuracy = [2, 2, 2, 2, 2, 2, 2]
    o_params = {"ga" : [80, 100, 7],
                "ghs" : [20, 600, 6, param_accuracy],#HMS, TotalIter, newHarm/Iter
                "sghs" : [20, 700, 6]}#HMS, TotalIter, newHarm/Iter
    a_params = {"ga" : [0.005,0.85,0.1,1,0.006,0.25,1.1,0.6,0.001],
                "ghs" : [0.85, 0.33],
                "sghs" : [0.85, 0.33, 0.01, 0.05, 0.01]}
    # %% Prepare works
    #choose metaheuristic
    alg = "ghs"
    #choose profiles
    iexps = [
             "1_10_12_6_12_0.5_3.8",
             "2_10_6_12_6_0.7_4",
             "3_15_12_6_12_0.55_4.2",
             "4_15_6_12_6_0.8_4"
            ]
    seeds = [19]
    n = 15
    works = {}
    k = 0
    for seed in seeds:
        for iexp in iexps:
            works[k]={"seed":seed, "iexp":iexp}
            k+=1
    # %% 
    min_vals = [50, 30, 30, 30, 0.1, 0.0, 2.5]
    max_vals = [250, 200, 200, 200, 0.45, 0.45, 5.5]
    for i in range(k):
        iexp = works[i]["iexp"]
        seed = works[i]["seed"]
        print(f"WORK {i}: Iexp {iexp}, n:{n}, seed:{seed}\n")
        sha_params = [n, 30, 0.5, 50.4, 40, fb[iexp], 7]
        oparams = o_params[alg]
        aparams = a_params[alg]#[0.85, 0.33]#[0.85, 0.33, 0.01, 0.05, 0.01]
        m = crease_he.Model(optim_params = oparams,#[12, 5, 7],
                            adapt_params = aparams,#[0.005,0.85,0.1,1,0.006,0.25,1.1,0.6,0.001], 
                            opt_algorithm = alg,
                            seed = seed,
                            offTime = offTime)
        #detailed explanations are needed to describe what each value in shape params means
        m.load_shape(shape='vesicle',
                    shape_params = sha_params,
                    minvalu = min_vals,
                    maxvalu = max_vals)
        #load target Iexp(q) IEXP_DATA  
        m.load_iq('./ICOMP_DATA/'+iexp+'.txt')
        m.solve(name = "s"+str(seed)+"_I"+iexp+"_n"+str(n)+"_w"+str(i),
                output_dir = './test_outputs',
                verbose = False,
                n_cores = 6)
        time.sleep(300)

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
        m.shut_down(t_shut_down)