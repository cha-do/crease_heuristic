import crease_he
import os

<<<<<<< Updated upstream
m = crease_he.Model(optim_params = [12, 73],#[12, 5, 7],
                 adapt_params = [0.9, 0.35],#[0.005,0.85,0.1,1,0.006,0.25,1.1,0.6,0.001], 
                opt_algorithm = "ghs")
=======
m = crease_he.Model(optim_params = [4, 4, 7],
                adapt_params = [0.005,0.85,0.1,1,0.006,0.25,1.1,0.6,0.001],
                opt_algorithm = "ga")
>>>>>>> Stashed changes
#detailed explanations are needed to describe what each value in shape params means
m.load_shape(shape='vesicle',shape_params=[24,54,0.5,50.4,50.4,0.55,7],
                                     minvalu = (50, 30, 30, 30, 0.1, 0.0, 0.1),
                                     maxvalu = (400, 200, 200, 200, 0.45, 0.45, 4))
                                     
m.load_iq('./IEXP_DATA/Itot_disper_10_Ain12_B6_Aout12_nLP7_dR0.2.txt')
<<<<<<< Updated upstream
#os.mkdir("./test_outputs_ga")
m.solve(name = 'jobPAR35e-2cicl73',output_dir='./test_outputs_ghs')
=======
os.mkdir("./test_outputs_34")
m.solve(output_dir='./test_outputs_34')
>>>>>>> Stashed changes
