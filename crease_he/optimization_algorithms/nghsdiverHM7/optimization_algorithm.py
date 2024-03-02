import numpy as np
import random
from os import path

class optimization_algorithm:
    """
    """

    def __init__(self,
                 optim_params = [10, 10, 1, None],
                 adapt_params = [0.1, 100, 3],
                 maxComputeTime = 10):
        self._name = "nghsdiverHM7"
        self._numadaptparams = 3
        self._numoptimparams = 4
        self.n_harmony = optim_params[0]
        self.n_iter = optim_params[1]
        self.harmsperiter = optim_params[2]
        self.pm = adapt_params[0]
        self.div = adapt_params[1]
        self.maxnumberdiv = adapt_params[2]
        self.param_accuracy = optim_params[3]
        self.bestfit = np.inf
        self.mct = maxComputeTime
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
        self.iter = iter
        if self.seed is not None:
            random.seed(int((((iter+1)*10)**2.5)%self.seed*((iter+1)*100)))
        improved = None
        imp = False
        Imp = False
        F1= open(self.address+'all_harmonies.txt','a')
        Fit = np.array(fit, dtype=float)
            
        if iter ==0:
            #First iteration
            self.harmony_fit = Fit
            self.compTimesHM = np.ones(self.n_harmony, dtype=int)
            self.worst_id = np.argmax(self.harmony_fit)
            self.best_id = np.argmin(self.harmony_fit)
            self.bestfit = self.harmony_fit[self.best_id]
            improved = self.best_id
            imp = True
            F1.write('#iter...all params...error...time...timeMach...IterTime\n')
            Iter = str(iter)
            for val in range(self.n_harmony): 
                #Save the params ofthe individual val
                F1.write(Iter+' ')
                for p in self.harmonies[val]:
                    F1.write(str(p)+' ')
                F1.write(str(self.harmony_fit[val])+' ')
                F1.write('%.2lf ' %(tic[val]))
                F1.write( '%.3lf %.3lf ' %(np.sum(tic), Tic)+'\n')
            F1.close()
        else:
            indexRepeted = []
            for i in range(len(self.new_harmony)):
                F1.write(str(iter)+' ')
                for p in self.new_harmony[i]:
                    F1.write(str(p)+' ')
                F1.write(str(fit[i])+' ')
                F1.write('%.2lf ' %(tic[i]))
                F1.write( '%.3lf %.3lf ' %(np.sum(tic), Tic)+'\n')
                #Update HM
                indextemp = np.all(self.harmonies == self.new_harmony[i], axis=1)
                if np.any(indextemp):
                    index = np.where(indextemp)[0][0]
                    indexRepeted.append(i)
                    if fit[i] < self.harmony_fit[index]:
                        self.harmony_fit[index] = fit[i]
                    self.compTimesHM[index] += 1
                    if self.compTimesHM[index] == self.mct:
                        if np.array_equal(self.tabuList,[[0,0,0,0,0,0,0]]):
                            self.tabuList[0] = self.harmonies[index]
                        else:
                            self.tabuList = np.vstack((self.tabuList, self.harmonies[index]))
                        with open(self.address+'tabuList.txt','a') as f:
                            for p in self.harmonies[index]:
                                f.write(str(p)+' ')
                            f.write('\n')
            F1.close()
            if len(indexRepeted) != 0:
                imp = True
                fit = np.delete(fit, indexRepeted, axis=0)
                self.new_harmony = np.delete(self.new_harmony, indexRepeted, axis=0)
                self.worst_id = np.argmax(self.harmony_fit)
                self.best_id = np.argmin(self.harmony_fit)
                self.bestfit = self.harmony_fit[self.best_id]
            
            for i in range(len(fit)):
                if fit[i] < self.harmony_fit[self.worst_id]:   
                    imp = True
                    self.harmonies[self.worst_id] = self.new_harmony[i]
                    self.harmony_fit[self.worst_id] = fit[i]
                    self.compTimesHM[self.worst_id] = 1
                    if fit[i] < self.bestfit:
                        if not Imp:
                            print('W{:d} Iteration: {:d}'.format(self.work, iter))
                            print("Fitness improved.\nOld best: {:.4f}".format(self.bestfit))
                        Imp = True
                        self.best_id = self.worst_id
                        self.bestfit = fit[i]
                    self.worst_id = np.argmax(self.harmony_fit)
                    
        if Imp:
            print("W{:d} New best: {:.4f}".format(self.work, self.bestfit))
            print('Generation best parameters '+str(self.harmonies[self.best_id]))
            improved = np.argmin(Fit)

        if imp:
            with open(self.address+'current_harmony_fit.txt', 'wb') as file:
                np.savetxt(file, self.harmony_fit)
            with open(self.address+'current_harmonies.txt', 'wb') as file:
                np.savetxt(file, self.harmonies)
            with open(self.address+'computeTimes.txt', 'wb') as file:
                np.savetxt(file, self.compTimesHM, fmt = '%d')
            f = open(self.address+'fitness_vs_gen.txt', 'a' )
            if iter == 0:
                f.write( 'Iter MiniHM MinHM AvgHM BestiHM BestHM CompTimesBestHM\n' )
            f.write( '%d ' %(iter) )
            f.write( '%d %.8lf ' %(self.worst_id,self.harmony_fit[self.worst_id]) )
            f.write( '%.8lf ' %(np.average(self.harmony_fit)) )
            f.write( '%d %.8lf %d' %(self.best_id, self.bestfit, self.compTimesHM[self.best_id]) )
            f.write( '\n' )
            f.close()
            #Save the individuals of the generation i in file results_i.txt
            F1= open(self.address+'results_'+str(iter)+'.txt','w')
            F1.write('#individual...all params...error...computedTimes\n')
            for val in range(self.n_harmony): 
                #Save the params ofthe individual val
                F1.write(str(val)+' ')
                for p in self.harmonies[val]:
                    F1.write(str(p)+' ')
                F1.write(str(self.harmony_fit[val])+' ')
                F1.write( '%d \n' %(self.compTimesHM[val]))
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
        self.compTimesHM = np.genfromtxt(self.address+'computeTimes.txt')
        if path.isfile(self.address+'tabuList.txt'):
            self.tabuList = np.genfromtxt(self.address+'tabuList.txt')#,dtype="float32")
            if np.array_equal(self.tabuList, []):
                self.tabuList = np.zeros((1,7))
            elif type(self.tabuList[0]).__name__ == 'float64':
                self.tabuList = np.array([self.tabuList])#, dtype = "float32")
        else:
            self.tabuList = np.zeros((1,7))
        iter = int(np.genfromtxt(self.address+'current_cicle.txt'))
        self.numberdiv = iter//self.div
        Tic = float(np.genfromtxt(self.address+'total_time.txt'))
        self.worst_id = np.argmax(self.harmony_fit)
        self.best_id = np.argmin(self.harmony_fit)
        self.bestfit = self.harmony_fit[self.best_id]
        self.new_harmony = np.genfromtxt(self.address+'current_new_harmony.txt')#,dtype="float32")
        if np.array_equal(self.new_harmony,[]):
            if self.seed is not None:
                random.seed(int((((iter)*10)**2.5)%self.seed*((iter)*100)))
            self.iter = self.div+1
            self._new_harmony()
        elif type(self.new_harmony[0]).__name__ == 'float64':
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
        fi.write( '\nTotalIters: %d' %(self.n_iter) )
        fi.write( '\nPm: %.4lf' %(self.pm) )
        fi.write( '\nDiv: %.4lf' %(self.div) )
        fi.write( '\nMaxNumberDiv: %d' %(self.maxnumberdiv) )
        fi.write( '\nHPI: %d' %(self.harmsperiter) )
        fi.write( '\nMCT: %d' %(self.mct) )
        if self.param_accuracy is not None:
            fi.write( f'\nParams accuracy: {self.param_accuracy}' )
        fi.close()
        self.harmonies = np.zeros((self.n_harmony,self.numvars))
        self.tabuList = np.zeros((1,7))
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
        self.numberdiv = 0
        return self.harmonies
    
    def _new_harmony(self):
        if (self.iter == 0) or (self.iter % self.div != 0) or (self.numberdiv >= self.maxnumberdiv):
            self.new_harmony = np.zeros((self.harmsperiter, self.numvars))
            for k in range(self.harmsperiter):   
                itl = True
                while itl: #itl: in tabu list 
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
                    itl = np.any(np.all(self.tabuList == self.new_harmony[k], axis=1))
        else:
            self.numberdiv += 1
            self._diverHM()
    
    def _diverHM(self):
        self.tabuList = np.zeros((1,7))
        tempcomptimes = np.zeros(self.n_harmony)
        tempcomptimes[0] = self.compTimesHM[self.best_id]
        self.compTimesHM = tempcomptimes.copy()
        with open(self.address+'tabuList.txt','w') as f:
            if tempcomptimes[0] >= self.mct:
                self.tabuList[0] = self.harmonies[self.best_id]
                for p in self.harmonies[self.best_id]:
                    f.write(str(p)+' ')
                f.write('\n')
            else:
                f.write('')
        tempHM = np.ones((self.n_harmony,self.numvars))*-1
        tempHM[0] = self.harmonies[self.best_id].copy()
        self.harmonies = tempHM.copy()
        tempHMfit = np.ones(self.n_harmony)*np.inf
        tempHMfit[0] = self.harmony_fit[self.best_id]
        self.harmony_fit = tempHMfit.copy()
        self.new_harmony = np.zeros((self.n_harmony-1, self.numvars))
        for i in range(len(self.new_harmony)):
            harmony = []
            for j in range(self.numvars):
                newparam = random.uniform(self.minvalu[j],self.maxvalu[j])
                if self.param_accuracy is not None:
                    newparam = np.round(newparam, self.param_accuracy[j])
                harmony.append(newparam)
            self.new_harmony[i] = np.array(harmony)
        self.worst_id = np.argmax(self.harmony_fit)
        self.best_id = 0
        with open(self.address+'iterdiver.txt','a') as f:
            f.write(str(self.iter)+" "+str(self.n_harmony-1)+"\n")