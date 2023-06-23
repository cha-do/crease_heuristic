import crease_he
import os

m = crease_he.Model(optim_params = [10,4],
                adapt_params = [0.9,0.2,2,2],
                opt_algorithm = "pso")
#detailed explanations are needed to describe what each value in shape params means
m.load_shape(shape='vesicle',shape_params=[24,54,0.5,50.4,50.4,0.55,7],
                                     minvalu = (50, 30, 30, 30, 0.1, 0.0, 0.1),
                                     maxvalu = (400, 200, 200, 200, 0.45, 0.45, 4))
                                     
m.load_iq('./IEXP_DATA/Itot_disper_10_Ain12_B6_Aout12_nLP7_dR0.2.txt')
os.mkdir("./test_outputs_42")
m.solve(output_dir='./test_outputs_42')


