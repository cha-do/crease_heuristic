# %% Imports
import crease_he
import os
import datetime, time

if __name__ == "__main__":
    offTime = None
    #offTime = datetime.datetime(2023, 11, 28, 6, 0)
    remainOn = False
    #os.mkdir("./test_outputs")
    fb = {"1_10_12_6_12_0.5" : 0.5,
          "2_10_6_12_6_0.7" : 0.7,
          "3_15_12_6_12_0.55" : 0.55,
          "4_15_6_12_6_0.8" : 0.8,
          "1_10_12_6_12_0.5_3.8" : 0.5,
          "2_10_6_12_6_0.7_4" : 0.7,
          "3_15_12_6_12_0.55_4.2" : 0.55,
          "4_15_6_12_6_0.8_4" : 0.8
          }
    # sh_params = {"10_Ain12_B6_Aout12_nLP7_dR0.2" : [28, 46, 0.7198, 50.4,  41.1713, 0.5779, 7], #
    #              "10_Ain12_B12_Aout12_nLP7_dR0.2" : [100,120,120,120,0.2,0.2,7],
    #              "20_Ain6_B12_Aout6_nLP7_dR0.2" : [15, 65, 0.6352, 50.4, 54.2969, 0.7840, 7]}
    #              "30_Ain12_B6_Aout12_nLP7_dR0.2" : []}
    # bg={"10_Ain12_B6_Aout12_nLP7_dR0.2" : [2.4,3.6], #
    #     #"10_Ain12_B12_Aout12_nLP7_dR0.2" : [3.0,3.8], #[2.8,4.0], #
    #     "20_Ain6_B12_Aout6_nLP7_dR0.2" : [2.8,4.0]} #
    #     #"30_Ain12_B6_Aout12_nLP7_dR0.2" : [2.7,3.9]}
    # Iexps = {"10_12_6_12" : "10_Ain12_B6_Aout12_nLP7_dR0.2", #
    #         "10_12_12_12" : "10_Ain12_B12_Aout12_nLP7_dR0.2", #
    #         "20_6_12_6" : "20_Ain6_B12_Aout6_nLP7_dR0.2", #
    #         "30_12_6_12" : "30_Ain12_B6_Aout12_nLP7_dR0.2"}
    o_params = {"ga" : [80, 70, 7],
                "ghs" : [20, 700, 6],#HMS, TotalIter, newHarm/Iter
                "sghs" : [20, 700, 6]}#HMS, TotalIter, newHarm/Iter
    a_params = {"ga" : [0.005,0.85,0.1,1,0.006,0.25,1.1,0.6,0.001],
                "ghs" : [0.85, 0.33],
                "sghs" : [0.85, 0.33, 0.01, 0.05, 0.01]}
    # %% Prepare works
    #choose metaheuristic
    alg = "ga"
    #choose profiles
    iexps = [#"3_15_12_6_12_0.55",
    #          "4_15_6_12_6_0.8",
    #          "1_10_12_6_12_0.5",
    #          "2_10_6_12_6_0.7",
            #  "3_15_12_6_12_0.55_4.2", #bg
            #  "4_15_6_12_6_0.8_4", #bg
            #  "1_10_12_6_12_0.5_3.8", #bg
             "2_10_6_12_6_0.7_4"] #bg
    seeds = [21]
    ns = [13]
    works = {}
    k = 0
    for seed in seeds:
        for n in ns:
            for iexp in iexps:
                works[k]={"seed":seed, "iexp":iexp, "n":n}
                k+=1
    # %% 
    min_vals = [50, 30, 30, 30, 0.1, 0.0, 2.5]
    max_vals = [250, 200, 200, 200, 0.45, 0.45, 4.5]
    for i in range(k):
        iexp = works[i]["iexp"]
        seed = works[i]["seed"]
        n = works[i]["n"]
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