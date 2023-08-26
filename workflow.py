import crease_he
import os

o_params = [3, 5]
a_params = [0.9, 0.35]
m = crease_he.Model(optim_params = o_params,#[12, 5, 7],
                 adapt_params = a_params,#[0.005,0.85,0.1,1,0.006,0.25,1.1,0.6,0.001], 
                opt_algorithm = "ghs",
                seed = 1)
#detailed explanations are needed to describe what each value in shape params means
m.load_shape(shape='vesicle',
             shape_params=[24,54,0.5,50.4,50.4,0.55,7],
             minvalu = (50, 30, 30, 30, 0.1, 0.0, 0.1),
             maxvalu = (400, 200, 200, 200, 0.45, 0.45, 4))
#load target Iexp(q)
Iexp = "10_Ain12_B6_Aout12_nLP7_dR0.2"                                     
m.load_iq('./IEXP_DATA/Itot_disper_'+Iexp+'.txt')
#os.mkdir("./test_outputs")
m.solve(name="Iexp"+Iexp+"_OptP"+str(a_params)+"_AdapP"+str(o_params),output_dir='./test_outputs')