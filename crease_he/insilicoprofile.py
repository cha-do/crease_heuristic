import os
import random
import numpy as np
from importlib import import_module
from crease_he.exceptions import CgaError
import matplotlib.pyplot as plt

class InSilicoProfile:
    def __init__(self, shape='vesicle',
                 shape_params=[24,54,0.5,50.4,50.4,0.55,7]):
        
        builtin_shapes=["vesicle","micelle","NP-solution","binary-NP-assembly",
                        "sffibril"]
        self.shape = shape
        if shape in builtin_shapes:
            sg = import_module('crease_he.shapes.'+shape+'.scatterer_generator')
            sg = sg.scatterer_generator
            print('Imported builtin shape {}\n'.format(shape))
            self.shape = shape
        else:
            from crease_he.plugins import plugins
            if shape in plugins.keys():
                sg = plugins[shape].load()
                print('Imported shape {} as a plugin'.format(shape))
            else:
                raise CgaError('Currently unsupported shape {}'.format(shape))
        self.scatterer_generator = sg(shape_params)

    def genprofile(self, params=[100, 120, 60, 120, 0.2, 0.2, 2.93],
                   q_min= 0.003, q_max= 0.1, q_vals=50,
                   output_dir='./ICOMP_DATA/',
                   seed = None,
                   plot = False,
                   n_cores = 1):
        q_range = np.linspace(q_min, q_max, num = q_vals)
        if seed is not None:
            random.seed(int(seed*7/3))
            np.random.seed(random.randint(seed*10, seed*10000))
        IQid = self.scatterer_generator.calculateScattering(q_range,[params],output_dir,n_cores)
        name = self.shape
        for par in params:
            name = name+"_"+str(par)
        IQid = np.array([q_range,IQid[0]])
        print(IQid)
        np.savetxt(output_dir+f'{name}.txt',np.c_[IQid.T])
        print("\nProfile generated whit params:",params,"\nAvalable in: ",output_dir+f'{name}.txt')

        if plot:
            figsize=(4,4)
            fig, ax = plt.subplots(figsize=(figsize))
            ax.plot(IQid[0],IQid[1],color='k',linestyle='-',ms=8,linewidth=1.3,marker='o')
            plt.xlim(q_range[0],q_range[-1])
            plt.ylim(2*10**(-5),20)
            plt.xlabel(r'q, $\AA^{-1}$',fontsize=20)
            plt.ylabel(r'$I$(q)',fontsize=20)
            plt.title(name)
            ax.set_xscale("log")
            ax.set_yscale("log")
            plt.show()

        
