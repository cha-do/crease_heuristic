from os import path
import numpy as np
import random

class optimization_algorithm:
    """
        Automatic diversification  based in the GDMSSE value, the HM is updated using the first HM
    """

    def __init__(self,
                 optim_params = [10, 10, 1, None],
                 adapt_params = [0.1, 0.5]):
        self._name = "nghsdiverHM6"
        self._numadaptparams = 2
        self._numoptimparams = 4
        self.n_harmony = optim_params[0]
        self.n_iter = optim_params[1]
        self.harmsperiter = optim_params[2]
        self.pm = adapt_params[0]
        self.threshold = adapt_params[1]
        self.div = self.n_harmony
        self.param_accuracy = optim_params[3]
        self.bestfit = np.inf
        self.seed = None
        self.work = None

    @property
    def numadaptparams(self):
        return self._numadaptparams
    
    @property
    def numoptimparams(self):
        return self._numoptimparams
    
    @property
    def name(self):
        return self._name
    
    def boundaryvalues(self, minvalu, maxvalu):
        self.minvalu = np.array(minvalu, dtype= float)
        self.maxvalu = np.array(maxvalu, dtype= float)
        self.numvars = len(minvalu)
        self.new_harmony = np.zeros((self.harmsperiter, self.numvars))
    
    def update_pop(self, fit, iter, tic, Tic):
        if self.seed is not None:
            random.seed(int((((iter+1)*10)**2.5)%self.seed*((iter+1)*100)))
        improved = None
        imp = False
        Imp = False
        F1= open(self.address+'all_harmonies.txt','a')
        if iter == 0:
            #First iteration
            self.harmony_fit = np.array(fit, dtype=float)
            self.worst_id = np.argmax(self.harmony_fit)
            self.best_id = np.argmin(self.harmony_fit)
            self.bestfit = self.harmony_fit[self.best_id]
            improved = self.best_id
            imp = True
            F1.write('#iter...all params...error...time...timeMach...IterTime\n')
            Iter = str(iter)
            self.HM1 = self.harmonies.copy()
            self.HMfit1 = self.harmony_fit.copy()
            np.savetxt(self.address+'HMfit1.txt',np.c_[self.HMfit1])
            np.savetxt(self.address+'HM1.txt',np.c_[self.HM1])
            for val in range(self.n_harmony): 
                #Save the params ofthe individual val
                F1.write(Iter+' ')
                for p in self.harmonies[val]:
                    F1.write(str(p)+' ')
                F1.write(str(self.harmony_fit[val])+' ')
                F1.write('%.2lf ' %(tic[val]))
                F1.write( '%.3lf %.3lf ' %(np.sum(tic), Tic)+'\n')
        else:
            for i in range(len(self.new_harmony)):
                F1.write(str(iter)+' ')
                for p in self.new_harmony[i]:
                    F1.write(str(p)+' ')
                F1.write(str(fit[i])+' ')
                F1.write('%.2lf ' %(tic[i]))
                F1.write( '%.3lf %.3lf ' %(np.sum(tic), Tic)+'\n')
                #Update harmonies 
                if fit[i] < self.harmony_fit[self.worst_id]:
                    imp = True
                    self.harmonies[self.worst_id, :] = self.new_harmony[i]
                    self.harmony_fit[self.worst_id] = fit[i]
                    if fit[i] < self.bestfit:
                        if not Imp:
                            print('W{:d} Iteration: {:d}'.format(self.work, iter))
                            print("Fitness improved.\nOld best: {:.4f}".format(self.bestfit))
                        Imp = True
                        self.best_id = self.worst_id
                        self.bestfit = fit[i]
                    self.worst_id = np.argmax(self.harmony_fit)
        F1.close()
        if Imp:
            print("W{:d} New best: {:.4f}".format(self.work, self.bestfit))
            print('Generation best parameters '+str(self.harmonies[self.best_id]))
            improved = np.argmin(fit)

        if (iter > 0) and (iter % self.div == 0):
            imp = self._diverHM(iter)

        if imp:
            np.savetxt(self.address+'current_harmony_fit.txt',np.c_[self.harmony_fit])
            np.savetxt(self.address+'current_harmonies.txt',np.c_[self.harmonies])
            f = open(self.address+'fitness_vs_gen.txt', 'a' )
            if iter == 0:
                f.write( 'Iter TimeMachine Time Mini Min Avg Besti Best\n' )
            f.write( '%d ' %(iter) )
            f.write( '%d %.8lf ' %(self.worst_id,self.harmony_fit[self.worst_id]) )
            f.write( '%.8lf ' %(np.average(self.harmony_fit)) )
            f.write( '%d %.8lf' %(self.best_id,self.bestfit) )
            f.write( '\n' )
            f.close()
            #Save the individuals of the generation i in file results_i.txt
            F1= open(self.address+'results_'+str(iter)+'.txt','w')
            F1.write('#individual...all params...error\n')
            for val in range(self.n_harmony): 
                #Save the params ofthe individual val
                F1.write(str(val)+' ')
                for p in self.harmonies[val]:
                    F1.write(str(p)+' ')
                F1.write(str(self.harmony_fit[val])+'\n')
                F1.flush()
            F1.close()
        
        #Create new harmony
        self._new_harmony()

        with open(self.address+'current_cicle.txt', 'wb') as file:
            np.savetxt(file, [iter+len(self.new_harmony)], fmt = '%d')
        with open(self.address+'current_new_harmony.txt', 'wb') as file:
            np.savetxt(file, self.new_harmony)
        
        return self.new_harmony, improved

    def resume_job(self, address):
        self.address = address
        self.harmonies = np.genfromtxt(self.address+'current_harmonies.txt')#,dtype="float32")
        self.harmony_fit = np.genfromtxt(self.address+'current_harmony_fit.txt')
        self.HM1 = np.genfromtxt(self.address+'HM1.txt')#,dtype="float32")
        self.HMfit1 = np.genfromtxt(self.address+'HMfit1.txt')
        iter = int(np.genfromtxt(self.address+'current_cicle.txt'))
        Tic = float(np.genfromtxt(self.address+'total_time.txt'))
        self.worst_id = np.argmax(self.harmony_fit)
        self.best_id = np.argmin(self.harmony_fit)
        self.bestfit = self.harmony_fit[self.best_id]
        self.new_harmony = np.genfromtxt(self.address+'current_new_harmony.txt')#,dtype="float32")
        if type(self.new_harmony[0]).__name__ == 'float64':
            self.new_harmony = np.array([self.new_harmony])#, dtype = "float32")
        print('W{:d} Restarting from iteration #{:d}'.format(self.work, iter))
        return iter, self.new_harmony, Tic
    
    def new_job(self, address):
        '''
        Produce a generation of (binary) chromosomes.
        
        Parameters
        ----------
        popnumber: int
            Number of individuals in a population.
        nloci: int
            Number of binary bits to represent each parameter in a chromosome.
        numvars: int
            Number of parameters in a chromosome.
            
        Returns
        -------
        pop: np.array of size (`popnumber`,`nloci`*`numvars`)
            A numpy array of binary bits representing the entire generation, 
            with each row representing a chromosome.
        '''
        self.address = address

        fi = open(address+'info.txt', 'a' )
        fi.write( '\nHMS: %d' %(self.n_harmony) )
        fi.write( '\nTotalIter: %d' %(self.n_iter) )
        fi.write( '\nPm: %.4lf' %(self.pm) )
        fi.write( '\nGDMSSE Threshold: %.3lf' %(self.threshold) )
        fi.write( '\nHPI: %d' %(self.harmsperiter) )
        if self.param_accuracy is not None:
            fi.write( f'\nParams accuracy: {self.param_accuracy}' )
        fi.close()
        self.harmonies = np.zeros((self.n_harmony,self.numvars))
        for i in range(self.n_harmony):
            harmony = []
            for j in range(self.numvars):
                newparam = random.uniform(self.minvalu[j],self.maxvalu[j])
                if self.param_accuracy is not None:
                    newparam = np.round(newparam, self.param_accuracy[j])
                harmony.append(newparam)
            self.harmonies[i] = np.array(harmony)#, dtype="float32")
        print('W'+str(self.work)+' New run')
        self.harmony_fit = np.zeros(self.n_harmony)
        return self.harmonies
    
    def _new_harmony(self):
        self.new_harmony = np.zeros((self.harmsperiter, self.numvars))
        for k in range(self.harmsperiter):
            #Create new harmony
            for j in range(self.numvars):
                if random.random() < self.pm:
                    newparam = random.uniform(self.minvalu[j],self.maxvalu[j])
                else:
                    x_r = 2 * self.harmonies[self.best_id, j] - self.harmonies[self.worst_id, j]
                    if x_r < self.minvalu[j]:
                        x_r = self.minvalu[j]
                    elif x_r > self.maxvalu[j]:
                        x_r = self.maxvalu[j]
                    newparam = self.harmonies[self.worst_id, j] + random.random()*(x_r-self.harmonies[self.worst_id, j])
                if self.param_accuracy is not None:
                    newparam = np.round(newparam, self.param_accuracy[j])
                self.new_harmony[k,j] = newparam
            
    def _diverHM(self, iter):
        divavg = np.average(self.harmony_fit)
        divmin = self.harmony_fit[self.best_id]
        divmetric = divmin/divavg
        div = False
        if divmetric >= self.threshold:
            harmdeleted = random.randint(0,self.n_harmony-1)
            tempHM = np.ones((self.n_harmony,self.numvars))
            tempHM[0] = self.harmonies[self.best_id].copy()
            tempHM[1:] = np.delete(self.HM1, harmdeleted, axis=0)
            self.harmonies = tempHM.copy()
            tempHMfit = np.ones(self.n_harmony)
            tempHMfit[0] = self.harmony_fit[self.best_id]
            tempHMfit[1:] = np.delete(self.HMfit1, harmdeleted, axis=0)
            self.harmony_fit = tempHMfit.copy()
            self.worst_id = np.argmax(self.harmony_fit)
            self.best_id = 0
            with open(self.address+'iterdiver.txt','a') as f:
                f.write(str(iter)+" "+str(divmetric)+"\n")
            div = True
        return div