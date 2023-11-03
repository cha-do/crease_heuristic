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

    def genprofile(self, params=[[100, 120, 60, 120, 0.2, 0.2, 2.93]],
                   q_lim = None,
                   q_range = None,
                   output_dir= None,
                   seed = None,
                   plot = False,
                   n_cores = 1):
        IQid = None
        if q_lim is not None or q_range is not None:
            if q_lim is not None:
                q_range = np.linspace(q_lim[0], q_lim[1], num = q_lim[2])
            if seed is not None:
                random.seed(int(seed*7/3))
                np.random.seed(random.randint(seed*10, seed*10000))
            if output_dir is None:
                IQid = self.scatterer_generator.calculateScattering(q_range,params,output_dir,n_cores)
            else:
                IQid = self.scatterer_generator.calculateScattering(q_range,params,output_dir[0],n_cores)
                for i in range(len(params)):
                    name = output_dir[1]+"_"
                    for j in range(len(params[i])):
                        name = name+str(params[i][j])
                        if j != len(params[i])-1:
                            name = name+"_"
                    iq = np.array([q_range,IQid[0][i]])
                    np.savetxt(output_dir[0]+f'{name}.txt',np.c_[iq.T], fmt="%.8f")
                    print("\nProfile generated wiht params:",params[i],"\nAvalable in: ",output_dir[0]+f'{name}.txt')

            if plot:
                colors = plt.cm.coolwarm(np.linspace(0,1,len(params)))
                figsize=(4,4)
                fig, ax = plt.subplots(figsize=(figsize))
                for i in range(len(params)):
                    name = ""
                    for j in range(len(params[i])):
                        name = name+str(params[i][j])
                        if j != len(params[i])-1:
                            name = name+"_"
                    ax.plot(q_range,IQid[0][i],color=colors[i],linestyle='-',ms=8,linewidth=1.3,marker='.',label=name)
                plt.xlim(q_range[0],q_range[-1])
                plt.ylim(2*10**(-5),20)
                plt.xlabel(r'q, $\AA^{-1}$',fontsize=20)
                plt.ylabel(r'$I$(q)',fontsize=20)
                ax.legend()
                ax.set_xscale("log")
                ax.set_yscale("log")
                plt.show()
        else:
            print("ERROR: q values no specified, use for this \"q_lim\" = (q_min, q_max, q_vals) or \"q_range\":(array with all the q values).")
        return IQid, q_range

        
