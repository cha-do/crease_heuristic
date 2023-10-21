# %% Imports
import crease_he
import os
import datetime, time

if __name__ == "__main__":
    offTime = None
    #offTime = datetime.datetime(2023, 9, 4, 6, 0)
    os.mkdir("./test_outputs")
    sh_params={"10_Ain12_B6_Aout12_nLP7_dR0.2" : [28, 46, 0.7198, 50.4,  41.1713, 0.5779, 7], #
            #"10_Ain12_B12_Aout12_nLP7_dR0.2" : [100,120,120,120,0.2,0.2,7], #
                "20_Ain6_B12_Aout6_nLP7_dR0.2" : [15, 65, 0.6352, 50.4, 54.2969, 0.7840, 7]} #
            #"30_Ain12_B6_Aout12_nLP7_dR0.2" : []}
    bg={"10_Ain12_B6_Aout12_nLP7_dR0.2" : [2.4,3.6], #
        #"10_Ain12_B12_Aout12_nLP7_dR0.2" : [3.0,3.8], #[2.8,4.0], #
        "20_Ain6_B12_Aout6_nLP7_dR0.2" : [2.8,4.0]} #
        #"30_Ain12_B6_Aout12_nLP7_dR0.2" : [2.7,3.9]}
    Iexps = {"10_12_6_12" : "10_Ain12_B6_Aout12_nLP7_dR0.2", #
            "10_12_12_12" : "10_Ain12_B12_Aout12_nLP7_dR0.2", #
            "20_6_12_6" : "20_Ain6_B12_Aout6_nLP7_dR0.2", #
            "30_12_6_12" : "30_Ain12_B6_Aout12_nLP7_dR0.2"}
    o_params = {"ga" : [80, 100, 7],
                "ghs" : [20, 7981],
                "sghs" : [20, 7981]}
    a_params = {"ga" : [0.005,0.85,0.1,1,0.006,0.25,1.1,0.6,0.001],
                "ghs" : [0.85, 0.33],
                "sghs" : [0.85, 0.33, 0.01, 0.05, 0.01]}
    # %% Prepare works
    #choose metaheuristic
    alg = "ga"
    #choose profiles
    iexps = ["10_12_6_12",
            #"10_12_12_12",
            "20_6_12_6"]
            #"30_12_6_12"]
    seeds = [1, 2, 3, 4, 5]
    works = {}
    k = 0
    for seed in seeds:
        for iexp in iexps:
            works[k]={"seed":seed,"iexp":iexp}
            k+=1
    # %% 
    for i in range(k):
        iexp = works[i]["iexp"]
        seed = works[i]["seed"]
        Iexp = Iexps[iexp]
        print(f"WORK {i}: Iexp {Iexp}, seed:{seed}\n")
        min_vals = [50, 30, 30, 30, 0.1, 0.0]
        max_vals = [400, 200, 200, 200, 0.45, 0.45]
        min_vals.append(bg[Iexp][0])
        max_vals.append(bg[Iexp][1])
        sha_params = sh_params[Iexp]
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
        m.load_iq('./IEXP_DATA/Itot_disper_'+Iexp+'.txt')
        m.solve(name=iexp+"_OptP"+str(aparams)+"_AdapP"+str(oparams),output_dir='./test_outputs',verbose=False)
        time.sleep(600)

    # %% Shutt down
    if offTime is not None:
        t_shut_down=10
        m.shut_down(str(t_shut_down))