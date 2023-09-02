import numpy as np
from os import path
import os
#from crease_he import utils
import random
import matplotlib
#matplotlib.use('Agg') ## uncomment this when running on cluster, comment out this line if on local
import matplotlib.pyplot as plt
import sys
from importlib import import_module
import time
import datetime
from warnings import warn
from crease_he.exceptions import CgaError

class Model:
    """
    The basic class that defines the model to be used to solve for a scattering
    profile.
    
    Attributes
    ----------
    pop_number: int. 
        Number of individuals within a generation.
    generations: int.
        Number of generations to run.
    nloci: int.
        Number of binary bits to represent each parameter in an individual.
        The decimal parameter value is converted to binary, with "all 0s"
        corresponding to the min value and "all 1s" corresponding to the
        max value. The larger the value, the finer the resultion for each parameter.
    adaptation_params: crease_he.adaptation_params.adaptation_params
        Object of adaptation parameters used for this model.

    See also
    --------
    crease_he.adaptaion_params.adaptation_params
    """

       
    def __init__(self,
                 optim_params = None,
                 adapt_params = None,
                 opt_algorithm = "ga",
                 seed = None,
                 offTime = None,
                 yaml_file='x'):
        if path.isfile(yaml_file):
            pass
            #TODO: populate all input parameters with input from yaml files
        else:
            builtin_opt_algorithm=["ga","pso","ghs","sghs"]
            if opt_algorithm in builtin_opt_algorithm:
                oa = import_module('crease_he.optimization_algorithms.'+opt_algorithm+'.optimization_algorithm')
                oa = oa.optimization_algorithm
                print('Imported builtin optimization algorithm {}\n'.format(opt_algorithm))
            else:
                raise CgaError('Currently unsupported optimization algorithm {}'.format(opt_algorithm))
            
            if adapt_params == None and optim_params != None:
                warn("Unspecified adaptation params. Fall back to the default values"
                    "of the {} algorithm.\n".format(opt_algorithm),stacklevel = 2)
                self.optimization_algorithm = oa(optim_params)
            elif adapt_params != None and optim_params == None:
                warn("Unspecified optimization params. Fall back to the default values"
                    "of the {} algorithm.\n".format(opt_algorithm),stacklevel = 2)
                self.optimization_algorithm = oa(adapt_params)
            else:
                self.optimization_algorithm = oa(optim_params, adapt_params)
            self.totalcicles = optim_params[1]
            self.seed = seed
            if seed is not None:
                random.seed(int(seed*7/3))
                np.random.seed(random.randint(seed*10, seed*10000))
            self.offTime = offTime
            if offTime is not None:
                print("Shutting down time setted at",offTime)
                print("Shutting down at",datetime.datetime.now())
                print("Shutting down in",offTime-datetime.datetime.now())
    
    def load_shape(self,shape="vesicle", shape_params=None,minvalu=None,maxvalu=None): 
        '''
        Load a shape.

        Parameters
        ----------
        shape: str. name of the shape.
            Currently supported builtin shapes are "vesicle" and "micelle". Can
            also specify a shape developed in a crease_he plugin.
        shape_params: list.
            Values of shape-specific descriptors. See the API of corresponding
            shape for details. If not specified, or an incorrect number of
            shape descriptor values are specified, the default values of the
            shape-specific descriptors will be loaded.
        minvalu,maxvalu: list.
            Values of the minimum and maximum boundaries of the
            parameters to be fit. If not specified, or an incorrect number of
            input parameter boundaries are specified, the default boundaries of
            the input parameters of the shape will be loaded.
        '''
        builtin_shapes=["vesicle","micelle","NP-solution","binary-NP-assembly",
                        "sffibril"]
        if shape in builtin_shapes:
            sg = import_module('crease_he.shapes.'+shape+'.scatterer_generator')
            sg = sg.scatterer_generator
            print('Imported builtin shape {}\n'.format(shape))
        else:
            from crease_he.plugins import plugins
            if shape in plugins.keys():
                sg = plugins[shape].load()
                print('Imported shape {} as a plugin'.format(shape))
            else:
                raise CgaError('Currently unsupported shape {}'.format(shape))
        
        #TODO: Complete the checker
        if shape_params is None:
            self.scatterer_generator = sg()
        elif minvalu == None or maxvalu == None:
            warn("Unspecified minimum and/or maximum parameter boundaries. Fall back to the default minimum "
                 "and maximum parameter boundaries of shape {}.\n".format(shape),stacklevel = 2)
            self.scatterer_generator = sg(shape_params)
            print("minimum parameter boundaries have been set to {},\n"
                  "maximum parameter boundaries have been set to {}.\n".format(
                   self.scatterer_generator.minvalu,
                   self.scatterer_generator.maxvalu))

        elif sg().numvars != len(minvalu) or sg().numvars != len(maxvalu):
               
            raise CgaError("Number of parameters in minvalu and/or maxvalu is not equal to number of parameters "
                 "required by shape {}.\n Shape {} requires {:d} parameters.\nminvalu has {:d} parameters.\n"
                 "maxvalu has {:d} parameters.".format(shape,shape,sg().numvars,
                                                     len(minvalu),len(maxvalu))) 
        else:
             self.scatterer_generator = sg(shape_params,minvalu,maxvalu)

        self.optimization_algorithm.boundaryvalues(self.scatterer_generator.minvalu, self.scatterer_generator.maxvalu)
            
            
            
    def load_iq(self,input_file_path,q_bounds=None):
        """
        Load an experimental I(q) profile [Iexp(q)] to the model, so that it can be
        solved later using "Model.solve".
        
        Parameters
        ----------
        input_file_path: str. Path to the input file. 
            The file should be organized in up to three column, with q-values in the first column, 
            corresponding I(q) values in the second, and optionally, corresponding error for the 
            I(q) in the third.
        q_bounds: [min,max].
            Define the minimum and maximum bound of the q region of interest. Any
            q-I(q) pairs outside of the defined bounds will be ignored during the
            fitting.
    
        See also
        --------
            crease_he.Model.solve()
        """
        loadvals = np.genfromtxt(input_file_path)
        self.qrange_load = loadvals[:,0]
        IQin_load = loadvals[:,1]
        if len(loadvals.T)>2:
            IQerr_load = loadvals[:,2]
            IQerr_load = np.true_divide(IQerr_load,np.max(IQin_load))
        else:
            IQerr_load = None
        self.IQin_load=np.true_divide(IQin_load,np.max(IQin_load))
        #TODO: highQ and lowQ needs to be able to be dynamically set
        if q_bounds is None:
            self.qrange = self.qrange_load
            self.IQin = self.IQin_load
            self.IQerr = IQerr_load
        else:
            lowQ = q_bounds[0]
            highQ = q_bounds[1]
            self.IQin = self.IQin_load[ np.where(self.qrange_load>=lowQ)[0][0]:np.where(self.qrange_load<=highQ)[0][-1] +1]
            if IQerr_load is None:
                self.IQerr = None
            else:
                self.IQerr = IQerr_load[ np.where(self.qrange_load>=lowQ)[0][0]:np.where(self.qrange_load<=highQ)[0][-1] +1]
            self.qrange = self.qrange_load[ np.where(self.qrange_load>=lowQ)[0][0]:np.where(self.qrange_load<=highQ)[0][-1] +1]
            
        baseline = self.IQin[0]
        if self.IQerr is not None:
            self.IQerr = np.true_divide(self.IQerr,baseline)
        self.IQin = np.true_divide(self.IQin,baseline)

        
    def solve(self,name = 'job',
              verbose = True,
              backend = 'debye',
              fitness_metric = 'log_sse',
              output_dir='./',
              n_cores=1,
              needs_postprocess = False):
        '''
        Fit the loaded target I(q) for a set of input parameters that maximize
        the fitness or minimize the error metric (fitness_metric).

        Parameters
        ----------
        name: str.
            Title of the current run. A folder of the name will be created
            under current working directory (output_dir), and all output files
            will be saved in that folder.
        verbose: bool. Default=True.
            If verbose is set to True, a figure will be produced at the end of
            each run, plotting the I(q) resulting from the best
            individual in the current generation and the target I(q).

            Useful for pedagogical purpose on jupyter notebook.
        fitness_metric: string. Default='log_sse'.
            The metric used to calculate fitness. Currently supported:
                "log_sse", sum of squared log10 difference at each q
                point.
        output_dir: string. Default="./" 
            Path to the working directory.
        '''
        ### checking if starting new run or restarting partial run
        name = self.optimization_algorithm.name+"_"+name+'_seed'+str(self.seed)
        address = output_dir+'/'+name+'/'
        if path.isfile(address+'current_cicle.txt'):
            currentcicle, pop, self.totalTime =self.optimization_algorithm.resume_job(address)
            fi = open(address+'info.txt', 'a' )
            fi.write( '\nSeed: ' )
            if self.seed is not None:
                fi.write( '%d' %(self.seed) )
            else:
                fi.write( '-1' )
            fi.close()
             # read in best iq for each generation
            bestIQ = np.genfromtxt(address+'best_iq.txt')
            # do not include q values in bestIQ array
            bestIQ = bestIQ[1:,:]
        else:
            self.totalTime = 0
            os.mkdir(address)
            currentcicle = 0
            pop = self.optimization_algorithm.new_job(address)
            # save best iq for each generation (plus q values)
            with open(address+'best_iq.txt','w') as f:
                np.savetxt(f,self.qrange,fmt="%-10f",newline='')
            bestIQ = [[]]

        Tic = time.time()
        for cicle in range(currentcicle, self.totalcicles):    
            print('\nIteration: {}'.format(cicle+1))
            if backend == 'debye':
                IQids = self.scatterer_generator.calculateScattering(self.qrange,pop,address,n_cores)
                fit=np.zeros(len(pop))
                for val in range(len(pop)):
                    ### calculate computed Icomp(q) ###
                    IQid=IQids[val]
                    err = self.fitness(IQid, fitness_metric)
                    fit[val] = err
                elitei=np.argmin(fit)
            
            tic=time.time()-Tic
            print('\nIteration time: {:.3f}s'.format(tic))
            pop, improved = self.optimization_algorithm.update_pop(fit, cicle, tic)
            
            #save new best IQ
            if improved:
                if np.array_equal(bestIQ,[[]]):
                    bestIQ[0] = IQids[elitei]
                else:
                    bestIQ = np.vstack((bestIQ, IQids[elitei]))
                with open(address+'best_iq.txt','a') as f:
                    f.write('\n')
                    np.savetxt(f,IQids[elitei],fmt="%-10f",newline='')

            if needs_postprocess:
                self.postprocess()

            if verbose and improved:
                    figsize=(4,4)
                    fig, ax = plt.subplots(figsize=(figsize))
                    ax.plot(self.qrange_load,self.IQin_load,color='k',linestyle='-',ms=8,linewidth=1.3,marker='o')
                    ax.plot(self.qrange,bestIQ[-1],color='b',linestyle='-',ms=8,linewidth=2)
                    plt.xlim(self.qrange[0],self.qrange[-1])
                    plt.ylim(2*10**(-5),20)
                    plt.xlabel(r'q, $\AA^{-1}$',fontsize=20)
                    plt.ylabel(r'$I$(q)',fontsize=20)
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    fig.savefig(address+'plot'+str(cicle)+'.png')
                    #plt.show()        
            
            if cicle == self.totalcicles-1:
                colors = plt.cm.coolwarm(np.linspace(0,1,len(bestIQ)))
                figsize=(4,4)
                fig, ax = plt.subplots(figsize=(figsize))
                ax.plot(self.qrange_load,self.IQin_load,color='k',linestyle='-',ms=8,linewidth=1.3,marker='o')
                for i in range(len(bestIQ)):
                    ax.plot(self.qrange,bestIQ[i],color=colors[i],linestyle='-',ms=8,linewidth=2)
                plt.xlim(self.qrange[0],self.qrange[-1])
                plt.ylim(2*10**(-5),20)
                plt.xlabel(r'q, $\AA^{-1}$',fontsize=20)
                plt.ylabel(r'$I$(q)',fontsize=20)
                ax.set_xscale("log")
                ax.set_yscale("log")
                plt.savefig(address+'iq_evolution.png',dpi=169,bbox_inches='tight')
            
            self.totalTime += time.time()-Tic
            Tic = time.time()
            np.savetxt(address+'total_time.txt',np.c_[self.totalTime])
            if self.offTime is not None:
                if datetime.datetime.now()>self.offTime:
                    t = 10
                    print("\nTime of shutt down")
                    self.shut_down(str(t))
                    time.sleep(t+10)
        print('Work ended.\nTotal time: {:.3f}s'.format(self.totalTime))
    
    def postprocess(self):
        #import weakref
        self.scatterer_generator.postprocess(self)
      
    def fitness(self, IQid, metric):
        err=0
        qfin=self.qrange[-1]
        for qi,qval in enumerate(self.qrange):
            if (IQid[qi]>0)&(self.IQin[qi]>0):
                if (qi<qfin):
                    wil=np.log(np.true_divide(self.qrange[qi+1],self.qrange[qi]))  # weighting factor
                else:
                    wil=np.log(np.true_divide(self.qrange[qi],self.qrange[qi-1]))  # weighting factor
                if metric == 'log_sse':
                    err+=wil*(np.log(np.true_divide(self.IQin[qi],IQid[qi])))**2  # squared log error 
            elif metric == 'chi2':
                if self.IQerr is None:
                    # chi^2 with weighting of IQin
                    err += np.true_divide(
                        np.square(self.IQin[qi]-IQid[qi]), np.square(self.IQin[qi]))
                else:
                    # chi^2 with weighting of IQerr
                    err += np.true_divide(
                        np.square(self.IQin[qi]-IQid[qi]), np.square(self.IQerr[qi]))
        return err
    
    def shut_down(self, t):
        print("\nSHUTTING DOWN.\n")
        os.system("shutdown /h")#s /f /t "+t)#h")
