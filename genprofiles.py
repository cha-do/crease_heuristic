# %% Impor crease package
import crease_he

# %% Load shape
sas = crease_he.InSilicoProfile(shape='vesicle',
                                shape_params=[24,54,0.5,50.4,50.4,0.55,7])

# %% Generate profile
sas.genprofile(params=[100, 120, 60, 120, 0.2, 0.2, 2.93],
               q_min= 0.003, q_max= 0.1, q_vals=50,
               output_dir='./ICOMP_DATA/',
               #seed = 1,
               plot = True)