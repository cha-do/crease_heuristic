import numpy as np
import random
import shutil

class optimization_algorithm:
    """
    """

    def __init__(self,
                 optim_params = [10, 10, 1, None],
                 adapt_params = [0.1, 0.5],
                 maxComputeTime = 10):
        self._name = "nghsdiverHM8"
        self._numadaptparams = 2
        self._numoptimparams = 4
        self.n_harmony = optim_params[0]
        self.n_iter = optim_params[1]
        self.harmsperiter = optim_params[2]
        self.pm = adapt_params[0]
        self.threshold = adapt_params[1]
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
        Fit = np.array(fit, dtype=float) 
        if iter == 0:
            #First iteration
            self.harmony_fit = Fit
            self.compTimesHM = np.ones(self.n_harmony, dtype=int)
            self.worst_id = np.argmax(self.harmony_fit)
            self.best_id = np.argmin(self.harmony_fit)
            self.bestfit = self.harmony_fit[self.best_id]
            improved = self.best_id
            imp = True
            with open(self.address+'currentState/all_harmonies.txt','a') as F1:
                F1.write('#iter...all params...error...time...timeMach...IterTime\n')
            for val in range(self.n_harmony): 
                #Save the params ofthe individual val
                self.all_harmonies_temp += str(iter)+' '
                for p in self.harmonies[val]:
                    self.all_harmonies_temp += str(p)+' '
                self.all_harmonies_temp += str(self.harmony_fit[val])+' '
                self.all_harmonies_temp += '%.2lf ' %(tic[val])
                self.all_harmonies_temp += '%.3lf %.3lf ' %(np.sum(tic), Tic)+'\n'
            with open(self.address+'currentState/best_evolution.csv','a') as f:
                f.write('Iter,R_core,t_Ain,t_B,t_Aout,s_Ain,sigma_R,log10(bg),bestSSE\n')
            self.best_evolution_temp += str(iter)+","
            for p in self.harmonies[self.best_id]:
                self.best_evolution_temp += str(p)+","
            self.best_evolution_temp += str(self.bestfit)
            self.best_evolution_temp += "\n"
            with open(self.address+'currentState/fitness_vs_gen.txt', 'a' ) as f:
                f.write( 'Iter MiniHM MinHM AvgHM BestiHM BestHM CompTimesBestHM\n' )
        else:
            indexRepeted = []
            for i in range(len(self.new_harmony)):
                self.all_harmonies_temp += str(iter)+' '
                for p in self.new_harmony[i]:
                    self.all_harmonies_temp += str(p)+' '
                self.all_harmonies_temp += str(fit[i])+' '
                self.all_harmonies_temp += '%.2lf ' %(tic[i])
                self.all_harmonies_temp += '%.3lf %.3lf ' %(np.sum(tic), Tic)+'\n'
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
                        for p in self.harmonies[index]:
                            self.tabuList_temp += str(p)+' '
                        self.tabuList_temp += '\n'
            if len(indexRepeted) != 0:
                imp = True
                fit = np.delete(fit, indexRepeted, axis=0)
                self.new_harmony = np.delete(self.new_harmony, indexRepeted, axis=0)
                self.worst_id = np.argmax(self.harmony_fit)
                self.best_id = np.argmin(self.harmony_fit)
                self.bestfit = self.harmony_fit[self.best_id]
                self.best_evolution_temp += str(iter)+","
                for p in self.harmonies[self.best_id]:
                    self.best_evolution_temp += str(p)+","
                self.best_evolution_temp += str(self.bestfit)+"\n"
            
            for i in range(len(fit)):
                if fit[i] < self.harmony_fit[self.worst_id]:   
                    imp = True
                    self.harmonies[self.worst_id, :] = self.new_harmony[i]
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
            self.best_evolution_temp += str(iter)+","
            for p in self.harmonies[self.best_id]:
                self.best_evolution_temp += str(p)+","
            self.best_evolution_temp += str(self.bestfit)+"\n"
        if imp:
            if not self.alreadydiv:
                self.gdm = self.bestfit/np.average(self.harmony_fit)
            self.fitness_vs_gen_temp +=  '%d ' %(iter)
            self.fitness_vs_gen_temp +=  '%d %.8lf ' %(self.worst_id,self.harmony_fit[self.worst_id])
            self.fitness_vs_gen_temp +=  '%.8lf ' %(np.average(self.harmony_fit))
            self.fitness_vs_gen_temp +=  '%d %.8lf %d\n' %(self.best_id, self.bestfit, self.compTimesHM[self.best_id])
            #Save the individuals of the generation i in file results_i.txt
            result = '#individual...all params...error...computedTimes\n'
            for val in range(self.n_harmony): 
                #Save the params ofthe individual val
                result += str(val)+' '
                for p in self.harmonies[val]:
                    result += str(p)+' '
                result += str(self.harmony_fit[val])+' '
                result += '%d \n' %(self.compTimesHM[val])
            self.results[iter] = result
        
        #Create new harmony
        self._new_harmony()

        return self.new_harmony, improved

    def saveinfo(self, totalTime, allIQ, bestIQ = None):
        address = self.address+"/currentState"
        shutil.copytree(address, address+"_temp", dirs_exist_ok=True)
        address += "/"
        with open(address+'current_cicle_temp.txt', 'w') as file:
            file.write(str(self.iter+len(self.new_harmony)))
        #Save the individuals of the generation i in file results_i.txt
        for iter in self.results.keys():
            with open(self.address+'results_'+str(iter)+'.txt','w') as F1:
                F1.write(self.results[iter])
        with open(address+'current_harmony_fit.txt', 'wb') as file:
            np.savetxt(file, self.harmony_fit)
        with open(address+'current_harmonies.txt', 'wb') as file:
            np.savetxt(file, self.harmonies)
        with open(address+'computeTimes.txt', 'wb') as file:
            np.savetxt(file, self.compTimesHM, fmt = '%d')
        with open(address+'current_new_harmony.txt', 'wb') as file:
            np.savetxt(file, self.new_harmony)
        with open(address+'all_iq.txt', 'a') as f:
            np.savetxt(f,allIQ)
        if bestIQ is not None:
            with open(address+'best_iq.txt', 'a') as f:
                np.savetxt(f,bestIQ)
        with open(address+'fitness_vs_gen.txt', 'a' ) as f:
            f.write(self.fitness_vs_gen_temp)
        with open(address+'all_harmonies.txt','a') as f:
            f.write(self.all_harmonies_temp)
        with open(address+'best_evolution.csv','a') as f:
            f.write(self.best_evolution_temp)
        with open(address+'tabuList.txt','a') as f:
            f.write(self.tabuList_temp)
        with open(address+'total_time.txt', 'w') as file:
            file.write(str(totalTime))
        with open(address+'current_cicle.txt', 'w') as file:
            file.write(str(self.iter+len(self.new_harmony)))
        self._restartSTR()

    def resume_job(self, address, deltaiter):
        from os import path, remove
        self.address = address
        address += "/currentState"
        flag = False
        try:
            iter = int(np.genfromtxt(address+'/current_cicle.txt'))
            iter_temp = int(np.genfromtxt(address+'/current_cicle_temp.txt'))
            flag = iter == iter_temp
        except Exception as e:
            flag = False
            print(f"W{self.work} Error:{e}")
        if not flag:
            shutil.copytree(address+"_temp", address, dirs_exist_ok=True)
            print(f"W{self.work}: Restarting from the temporal copy.")
        address += "/"
        self.harmonies = np.genfromtxt(address+'current_harmonies.txt')#,dtype="float32")
        self.harmony_fit = np.genfromtxt(address+'current_harmony_fit.txt')
        self.compTimesHM = np.genfromtxt(address+'computeTimes.txt')
        if path.isfile(address+'tabuList.txt'):
            self.tabuList = np.genfromtxt(address+'tabuList.txt')#,dtype="float32")
            if np.array_equal(self.tabuList, []):
                self.tabuList = np.zeros((1,self.numvars))
            elif type(self.tabuList[0]).__name__ == 'float64':
                self.tabuList = np.array([self.tabuList])#, dtype = "float32")
        else:
            self.tabuList = np.zeros((1,self.numvars))
        iter = int(np.genfromtxt(address+'current_cicle.txt'))
        for i in range(iter,iter+deltaiter+1):
            if path.isfile(self.address+'results_'+str(i)+'.txt'):
                remove(self.address+'results_'+str(i)+'.txt')
            if path.isfile(self.address+'plot'+str(i)+'.png'):
                remove(self.address+'plot'+str(i)+'.png')     
        Tic = float(np.genfromtxt(address+'total_time.txt'))
        self.worst_id = np.argmax(self.harmony_fit)
        self.best_id = np.argmin(self.harmony_fit)
        self.bestfit = self.harmony_fit[self.best_id]
        self.alreadydiv = path.isfile(address+'iterdiver.txt')
        if not self.alreadydiv:
            self.gdm = self.bestfit/np.average(self.harmony_fit)
        else:
            self.gdm = 0
        self.new_harmony = np.genfromtxt(address+'current_new_harmony.txt')#,dtype="float32")
        if type(self.new_harmony[0]).__name__ == 'float64':
            self.new_harmony = np.array([self.new_harmony])#, dtype = "float32")
        print('W{:d} Restarting from iteration #{:d}'.format(self.work, iter))
        self._restartSTR()
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

        with open(address+'info.txt', 'a' ) as fi:
            fi.write( '\nHMS: %d' %(self.n_harmony) )
            fi.write( '\nTotalIters: %d' %(self.n_iter) )
            fi.write( '\nPm: %.4lf' %(self.pm) )
            fi.write( '\nGDMSSE Threshold: %.3lf' %(self.threshold) )
            fi.write( '\nHPI: %d' %(self.harmsperiter) )
            fi.write( '\nMCT: %d' %(self.mct) )
            if self.param_accuracy is not None:
                fi.write( f'\nParams accuracy: {self.param_accuracy}' )
        self.harmonies = np.zeros((self.n_harmony,self.numvars))
        self.tabuList = np.zeros((1,self.numvars))
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
        self.gdm = 0
        self.alreadydiv = False
        self._restartSTR()
        return self.harmonies
    
    def _new_harmony(self):
        if self.gdm < self.threshold:
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
            self._diverHM()
    
    def _diverHM(self):
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
        with open(self.address+'currentState/iterdiver.txt','a') as f:
            f.write(str(self.iter)+" "+str(self.gdm)+"\n")
        self.gdm = 0
        self.alreadydiv = True
    
    def _restartSTR(self):
        self.results = {}
        self.fitness_vs_gen_temp = ""
        self.all_harmonies_temp = ""
        self.best_evolution_temp = ""
        self.tabuList_temp = ""