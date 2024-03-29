import numpy as np
import random

class optimization_algorithm:
    """
    """

    def __init__(self,
                 optim_params = [10, 10, 1],
                 adapt_params = [0.9, 0.99, 0.01, 0.05, 0.0001]):
        self._name = "sghs"
        self._numadaptparams = 3
        self._numoptimparams = 2
        self.n_harmony = optim_params[0]
        self.n_iter = optim_params[1]
        self.harmsperiter = optim_params[2]
        self.hmcr = adapt_params[0]
        self.par_max = adapt_params[1]
        self.par_min = adapt_params[2]
        self.bw_max = adapt_params[3]
        self.bw_min = adapt_params[4]
        self.bestfit = np.inf
        self.par = 0
        self.bw = 0

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
        self.new_harmony = np.zeros((self.harmsperiter,self.numvars))
    
    def update_pop(self, fit, iter, tic, Tic):
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
            F1.write('#iter...all params...error...time\n')
            Iter = str(iter)
            for val in range(self.n_harmony): 
                #Save the params ofthe individual val
                F1.write(Iter+' ')
                for p in self.harmonies[val]:
                    F1.write(str(p)+' ')
                F1.write('%.5lf' %(self.harmony_fit[val])+' ')
                F1.write('%.2lf ' %(tic[val])+'\n')
        else:
            for i in range(self.harmsperiter):
                F1.write(str(iter)+' ')
                for p in self.new_harmony[i]:
                    F1.write(str(p)+' ')
                F1.write('%.5lf' %(fit[i])+' ')
                F1.write('%.2lf ' %(tic[i])+'\n')
                #Update harmonies 
                if fit[i] < self.harmony_fit[self.worst_id]:   
                    imp = True
                    self.harmonies[self.worst_id, :] = self.new_harmony[i]
                    self.harmony_fit[self.worst_id] = fit[i]
                    if fit[i] < self.bestfit:
                        if not Imp:
                            print('Iteration: {:d}'.format(iter))
                            print("Fitness improved.\nOld best: {:.4f}".format(self.bestfit))
                        Imp = True
                        self.best_id = self.worst_id
                        self.bestfit = fit[i]
                    self.worst_id = np.argmax(self.harmony_fit)
        F1.close()
        if Imp:
            print("New best: {:.4f}".format(self.bestfit))
            print('Generation best parameters '+str(self.harmonies[self.best_id]))
            improved = np.argmin(fit)
        
        
        #Create new harmony
        self.par = self.par_min + ((self.par_max-self.par_min)/self.n_iter) * iter 
        
        if iter<self.n_iter/2:
            self.bw = self.bw_max - ((self.bw_max-self.bw_min)/self.n_iter) * 2 * iter
        else:
            self.bw = self.bw_min
        
        for k in range(self.harmsperiter):
            self.new_harmony[k] = self._new_harmony()

        if imp:
            np.savetxt(self.address+'current_harmony_fit.txt',np.c_[self.harmony_fit])
            np.savetxt(self.address+'current_harmonies.txt',np.c_[self.harmonies])
            f = open(self.address+'fitness_vs_gen.txt', 'a' )
            if iter == 0:
                f.write( 'iter mini min avg besti best\n' )
            f.write( '%d ' %(iter) )
            f.write( '%.3lf %.3lf ' %(np.sum(tic), Tic) )
            f.write( '%d %.8lf ' %(self.worst_id,self.harmony_fit[self.worst_id]) )
            f.write( '%.8lf ' %(np.average(self.harmony_fit)) )
            f.write( '%d %.8lf ' %(self.best_id,self.bestfit) )
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

        np.savetxt(self.address+'current_cicle.txt',np.c_[iter+1])
        np.savetxt(self.address+'current_new_harmony.txt',np.c_[self.new_harmony])

        return self.new_harmony, improved

    def resume_job(self, address):
        self.address = address
        self.harmonies = np.genfromtxt(self.address+'current_harmonies.txt')
        self.harmony_fit = np.genfromtxt(self.address+'current_harmony_fit.txt')
        self.new_harmony = np.genfromtxt(self.address+'current_new_harmony.txt')
        if type(self.new_harmony[0]).__name__ == 'float64':
            self.new_harmony = np.array([self.new_harmony])
        iter = int(np.genfromtxt(self.address+'current_cicle.txt'))
        self.par = self.par_min + ((self.par_max-self.par_min)/self.n_iter) * iter 
        if iter<self.n_iter/2:
            self.bw = self.bw_max - ((self.bw_max-self.bw_min)/self.n_iter) * 2 * iter
        else:
            self.bw = self.bw_min
        Tic = float(np.genfromtxt(self.address+'total_time.txt'))
        self.worst_id = np.argmax(self.harmony_fit)
        self.best_id = np.argmin(self.harmony_fit)
        self.bestfit = self.harmony_fit[self.best_id]
        print('Restarting from iteration #{:d}'.format(iter))
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
        fi.write( '\nHMS: ' )
        fi.write( '%d' %(self.n_harmony) )
        fi.write( '\nIter: ' )
        fi.write( '%d' %(self.n_iter) )
        fi.write( '\nHMCR: ' )
        fi.write( '%.2lf' %(self.hmcr) )
        fi.write( '\nbwMin: ' )
        fi.write( '%.2lf' %(self.bw_min) )
        fi.write( '\nbwMax: ' )
        fi.write( '%.2lf' %(self.bw_max) )
        fi.write( '\nPARMin: ' )
        fi.write( '%.2lf' %(self.par_min) )
        fi.write( '\nPARMax: ' )
        fi.write( '%.2lf' %(self.par_max) )
        fi.close()

        self.harmonies = np.zeros((self.n_harmony,self.numvars))
        for i in range(self.n_harmony):
            for j in range(self.numvars):
                self.harmonies[i][j] = random.uniform(self.minvalu[j],self.maxvalu[j])
        print('New run')
        self.harmony_fit = np.zeros(self.n_harmony)
        return self.harmonies
    
    def _new_harmony(self):
        #Create new harmony
        new_harmony = []
        for j in range(self.numvars):
            if random.random() < self.hmcr:
                if random.random() < self.par:
                    newparam = self.harmonies[self.best_id, j]
                else:
                    idx = random.randint(0,self.n_harmony-1)
                    rang = (self.maxvalu[j]-self.minvalu[j])/2
                    newparam= self.harmonies[idx, j] + random.uniform(-rang,rang) * self.bw
                    if newparam < self.minvalu[j]:
                        newparam = self.minvalu[j]
                    elif newparam > self.maxvalu[j]:
                        newparam = self.maxvalu[j]
            else:
                newparam = random.uniform(self.minvalu[j],self.maxvalu[j])
            new_harmony.append(newparam)
        return np.array(new_harmony)
