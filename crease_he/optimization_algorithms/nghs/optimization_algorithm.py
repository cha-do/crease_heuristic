import numpy as np
import random
import shutil

class optimization_algorithm:
    """
    Class for the NGHS (Novel Global Harmony Search) optimization algorithm.

    Attributes
    ----------
    hyperparameters : list of length 3, default=[40, 800, 0.07]
        A list containing the hyperparameters required for the NGHS algorithm.
        The list corresponds to [`HMS`, `NI`, `pm`].
        - HMS (Harmony Memory Size) : int, default=40
            The number of solution vectors stored in the Harmony Memory (HM).
        - NI (Number of Iterations) : int, default=800
            The total number of iterations the algorithm will run.
        - pm (Mutation Probability) : float, default=0.07
            The probability of a genetic mutation occurring.

    adapt_hyperparams : list, default=None
        Additional hyperparameters required for the execution of the optimization 
        algorithm. These parameters are not used in the NGHS algorithm.

    param_accuracy : list of length `numvars`, default=None
        A list defining the precision (number of decimal places) for each Shape parameter 
        in the optimization process. It should be a list of integers, where each entry 
        corresponds to the number of decimals to consider for each parameter.

    waitinglistSize : int, default=10
        The size of the waiting list, which stores information about solutions 
        that recently exited the HM. These solutions are retained in case a future 
        reevaluation determines they meet the criteria to re-enter the HM.

    maxComputeTime : int, default=10
        Maximum number of times a solution can be evaluated if it already appears 
        in the HM or waiting list, before it is added to the tabu list to prevent 
        further evaluations.
    """
    def __init__(self,
                 hyperparameters = [40, 800, 0.07],
                 adapt_hyperparams = None,
                 param_accuracy = None,
                 waitinglistSize = 10,
                 maxComputeTime = 10):
        self._name = "nghs"
        self._numadaptparams = 0
        self._numoptimparams = 3
        self.HMS = hyperparameters[0]
        self.NI = hyperparameters[1]
        self.pm = hyperparameters[2]
        self.param_accuracy = param_accuracy
        self.bestfit = np.inf
        self.wls = waitinglistSize
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
        """
        Set the boundaries and dimensionality (`numvars`) of the search space 
        for the Shape's parameters.

        Parameters
        ----------
        minvalu : list of length `numvars`
            A list containing the minimum values for each Shape parameter.
        maxvalu : list of length `numvars`
            A list containing the maximum values for each Shape parameter.

        Returns
        -------
        None
        """
        self.minvalu = np.array(minvalu, dtype= float)
        self.maxvalu = np.array(maxvalu, dtype= float)
        self.numvars = len(minvalu)
        self.new_harmony = np.zeros((1, self.numvars))
    
    def update_pop(self, fit, iter, tic, Tic):
        """
        Update the Harmony Memory (HM) with the results of the current iteration.

        Parameters
        ----------
        fit : list
            A list containing the fitness values of the solutions evaluated 
            during the current iteration.
        iter : int
            The current iteration number.
        tic : list
            A list containing the computation time (machine time) taken 
            to evaluate each solution in the current iteration.
        Tic : float
            The total computation time for the entire iteration.

        Returns
        -------
        new_harmony : ndarray of shape (1, `numvars`)
            The new solution to be evaluated in the next iteration.
        improved : int or None
            If this is in this iteration, the index of the new best solution
            from the list of evaluated solutions of this iteration. Otherwise,
            it returns None.
        """

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
            self.compTimesHM = np.ones(self.HMS, dtype=int)
            self.worst_id = np.argmax(self.harmony_fit)
            self.best_id = np.argmin(self.harmony_fit)
            self.bestfit = self.harmony_fit[self.best_id]
            self.worst_idWL = 0
            self.best_idWL = self.wls-1
            improved = self.best_id
            imp = True
            with open(self.address+'currentState/all_harmonies.txt','a') as F1:
                F1.write('#iter...all params...error...time\n')
            for val in range(self.HMS): 
                #Save the params ofthe individual val
                self.all_harmonies_temp += str(iter)+' '
                for p in self.HM[val]:
                    self.all_harmonies_temp += str(p)+' '
                self.all_harmonies_temp += str(self.harmony_fit[val])+' '
                self.all_harmonies_temp += '%.2lf ' %(tic[val])+'\n'
            # with open(self.address+'currentState/best_evolution.csv','a') as f:
            #     f.write('Iter,R_core,t_Ain,t_B,t_Aout,s_Ain,sigma_R,log10(bg),bestSSE\n')
            # self.best_evolution_temp += str(iter)+","
            # for p in self.HM[self.best_id]:
            #     self.best_evolution_temp += str(p)+","
            # self.best_evolution_temp += str(self.bestfit)
            # self.best_evolution_temp += "\n"
            with open(self.address+'currentState/fitness_vs_gen.txt', 'a' ) as f:
                f.write( 'Iter MiniWL MinWL AvgWL BestiWL BestWL CompTimesBestWL MiniHM MinHM AvgHM BestiHM BestHM CompTimesBestHM\n' )
        else:
            indexRepeted = []
            for i in range(len(self.new_harmony)):
                self.all_harmonies_temp += str(iter)+' '
                for p in self.new_harmony[i]:
                    self.all_harmonies_temp += str(p)+' '
                self.all_harmonies_temp += str(fit[i])+' '
                self.all_harmonies_temp += '%.2lf ' %(tic[i])+'\n'
                #Update HM
                indextemp = np.all(self.HM == self.new_harmony[i], axis=1)
                if np.any(indextemp):
                    index = np.where(indextemp)[0][0]
                    indexRepeted.append(i)
                    if fit[i] < self.harmony_fit[index]:
                        self.harmony_fit[index] = fit[i]
                    self.compTimesHM[index] += 1
                    if self.compTimesHM[index] == self.mct:
                        if np.array_equal(self.tabuList,[[0,0,0,0,0,0,0]]):
                            self.tabuList[0] = self.HM[index]
                        else:
                            self.tabuList = np.vstack((self.tabuList, self.HM[index]))
                        for p in self.HM[index]:
                            self.tabuList_temp += str(p)+' '
                        self.tabuList_temp += '\n'
                else:
                    indextemp = np.all(self.waitingList == self.new_harmony[i], axis=1)
                    if np.any(indextemp):
                        index = np.where(indextemp)[0][0]
                        indexRepeted.append(i)
                        if fit[i] < self.WL_fit[index]:
                            self.WL_fit[index] = fit[i]
                        self.compTimesWL[index] += 1
                        if self.compTimesWL[index] == self.mct:
                            if np.array_equal(self.tabuList,[[0,0,0,0,0,0,0]]):
                                self.tabuList[0] = self.waitingList[index]
                            else:
                                self.tabuList = np.vstack((self.tabuList, self.waitingList[index]))
                            for p in self.waitingList[index]:
                                self.tabuList_temp += str(p)+' '
                            self.tabuList_temp += '\n'
            if len(indexRepeted) != 0:
                imp = True
                fit = np.delete(fit, indexRepeted, axis=0)
                self.new_harmony = np.delete(self.new_harmony, indexRepeted, axis=0)
                self.worst_id = np.argmax(self.harmony_fit)
                self.best_idWL = np.argmin(self.WL_fit)
                while self.WL_fit[self.best_idWL]<self.harmony_fit[self.worst_id]:
                    tempharm = self.waitingList[self.best_idWL].copy()
                    self.waitingList[self.best_idWL] = self.HM[self.worst_id]
                    self.HM[self.worst_id] = tempharm
                    self.WL_fit[self.best_idWL], self.harmony_fit[self.worst_id] = self.harmony_fit[self.worst_id], self.WL_fit[self.best_idWL]
                    self.compTimesWL[self.best_idWL], self.compTimesHM[self.worst_id] = self.compTimesHM[self.worst_id], self.compTimesWL[self.best_idWL]
                    self.worst_id = np.argmax(self.harmony_fit)
                    self.best_idWL = np.argmin(self.WL_fit)
                    
                self.best_id = np.argmin(self.harmony_fit)
                self.bestfit = self.harmony_fit[self.best_id]
                self.worst_idWL = np.argmax(self.WL_fit)
                # self.best_evolution_temp += str(iter)+","
                # for p in self.HM[self.best_id]:
                #     self.best_evolution_temp += str(p)+","
                # self.best_evolution_temp += str(self.bestfit)+"\n"
            
            for i in range(len(fit)):
                if fit[i] < self.WL_fit[self.worst_idWL]:
                    imp = True
                    if fit[i] < self.harmony_fit[self.worst_id]:   
                        self.waitingList[self.worst_idWL] = self.HM[self.worst_id].copy()
                        self.WL_fit[self.worst_idWL] = self.harmony_fit[self.worst_id]
                        self.compTimesWL[self.worst_idWL] = self.compTimesHM[self.worst_id]
                        self.best_idWL = self.worst_idWL
                        self.HM[self.worst_id] = self.new_harmony[i]
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
                    else:
                        self.waitingList[self.worst_idWL] = self.new_harmony[i].copy()
                        self.WL_fit[self.worst_idWL] = fit[i]
                        self.compTimesWL[self.worst_idWL] = 1
                        if fit[i] < self.WL_fit[self.best_idWL]:
                            self.best_idWL = self.worst_idWL
                    self.worst_idWL = np.argmax(self.WL_fit)
        if Imp:
            print("W{:d} New best: {:.4f}".format(self.work, self.bestfit))
            print('Generation best parameters '+str(self.HM[self.best_id]))
            improved = np.argmin(Fit)
            # self.best_evolution_temp += str(iter)+","
            # for p in self.HM[self.best_id]:
            #     self.best_evolution_temp += str(p)+","
            # self.best_evolution_temp += str(self.bestfit)+"\n"
        if imp:
            self.fitness_vs_gen_temp +=  '%d ' %(iter)
            self.fitness_vs_gen_temp +=  '%d %.5lf ' %(self.worst_idWL,self.WL_fit[self.worst_idWL]) 
            self.fitness_vs_gen_temp +=  '%.5lf ' %(np.average(self.WL_fit)) 
            self.fitness_vs_gen_temp +=  '%d %.5lf %d ' %(self.best_idWL, self.WL_fit[self.best_idWL], self.compTimesWL[self.best_idWL]) 
            self.fitness_vs_gen_temp +=  '%d %.8lf ' %(self.worst_id,self.harmony_fit[self.worst_id]) 
            self.fitness_vs_gen_temp +=  '%.8lf ' %(np.average(self.harmony_fit)) 
            self.fitness_vs_gen_temp +=  '%d %.8lf %d' %(self.best_id, self.bestfit, self.compTimesHM[self.best_id]) 
            self.fitness_vs_gen_temp +=  '\n' 
            #Save the individuals of the generation i in file results_i.txt
            result = '#individual...all params...error...computedTimes\n'
            for val in range(self.HMS): 
                #Save the params ofthe individual val
                result += str(val)+' '
                for p in self.HM[val]:
                    result += str(p)+' '
                result += str(self.harmony_fit[val])+' '
                result += '%d \n' %(self.compTimesHM[val])
            for val in range(self.wls): 
                #Save the params ofthe individual val
                result += '-'+str((val))+' '
                for p in self.waitingList[val]:
                    result += str(p)+' '
                result += str(self.WL_fit[val])+' '
                result += '%d \n' %(self.compTimesWL[val])
            self.results[iter] = result
        
        #Create new harmony
        self._new_harmony()

        return self.new_harmony, improved

    def resume_job(self, address, deltaiter):
        """
        Resume execution from the files located in the specified folder.

        Parameters
        ----------
        address : str
            The relative path to the folder containing the execution information.
        deltainter : int
            The interval of iterations between saves of the current state of the 
            execution.

        Returns
        -------
        iter : int
            The last iteration of the execution saved in the files.
        new_harmony : ndarray of shape (1, `numvars`)
            The current new harmony to be evaluated.
        Tic : float
            The time in seconds that the execution has been running.
        """

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
        self.HM = np.genfromtxt(address+'current_harmonies.txt')#,dtype="float32")
        self.waitingList = self.HM[-self.wls:].copy()
        self.HM = self.HM[:-self.wls]
        self.harmony_fit = np.genfromtxt(address+'current_harmony_fit.txt')
        self.WL_fit, self.harmony_fit = self.harmony_fit[-self.wls:], self.harmony_fit[:-self.wls] 
        self.compTimesHM = np.genfromtxt(address+'computeTimes.txt')
        self.compTimesWL, self.compTimesHM = self.compTimesHM[-self.wls:], self.compTimesHM[:-self.wls] 
        self.new_harmony = np.genfromtxt(address+'current_new_harmony.txt')#,dtype="float32")
        if type(self.new_harmony[0]).__name__ == 'float64':
            self.new_harmony = np.array([self.new_harmony])#, dtype = "float32")
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
        self.best_idWL = np.argmin(self.WL_fit)
        self.worst_idWL = np.argmax(self.WL_fit)
        print('W{:d} Restarting from iteration #{:d}'.format(self.work, iter))
        self._restartSTR()
        return iter, self.new_harmony, Tic
    
    def new_job(self, address):
        """
        Starts a new execution by initializing the Harmony Memory (HM) and other 
        necessary variables, and creates the initial files containing information 
        about the execution.

        Parameters
        ----------
        address : str
            The relative path to the folder where the execution information will be
            saved.

        Returns
        -------
        HM : ndarray of shape (`HMS`, `numvars`)
            A numpy array representing the Harmony Memory, with each row corresponding 
            to a solution vector.
        """

        self.address = address

        with open(address+'info.txt', 'a' ) as fi:
            fi.write( '\nHMS: %d' %(self.HMS) )
            fi.write( '\nTotalIters: %d' %(self.NI) )
            fi.write( '\nPm: %.4lf' %(self.pm) )
            fi.write( '\nWLS: %d' %(self.wls) )
            fi.write( '\nMCT: %d' %(self.mct) )
            if self.param_accuracy is not None:
                fi.write( f'\nParams accuracy: {self.param_accuracy}' )
        self.HM = np.zeros((self.HMS,self.numvars))
        self.waitingList = np.ones((self.wls,self.numvars))*-1
        self.tabuList = np.zeros((1,self.numvars))
        for i in range(self.HMS):
            harmony = []
            for j in range(self.numvars):
                newparam = random.uniform(self.minvalu[j],self.maxvalu[j])
                if self.param_accuracy is not None:
                    newparam = np.round(newparam, self.param_accuracy[j])
                harmony.append(newparam)
            self.HM[i] = np.array(harmony)#, dtype="float32")
        print('W'+str(self.work)+' New run')
        self.WL_fit = np.ones(self.wls)*np.inf
        self.compTimesWL = np.zeros(self.wls, dtype=int)
        self.harmony_fit = np.zeros(self.HMS)
        self._restartSTR()
        return self.HM

    def saveinfo(self, totalTime, allIQ, bestIQ = None):
        """
        Save the current state of the optimization process to .txt files. 
        This is useful for analysis or to resume the process in case of an unexpected 
        interruption.

        Parameters
        ----------
        totalTime : float
            The total time that the execution has taken so far.
        allIQ : ndarray
            The scattering intensity of the evaluated solutions since the last time 
            this function was called.
        bestIQ : ndarray, default=None
            The scattering intensity of the best solutions found since the last time 
            this function was called.

        Returns
        -------
        None
        """
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
            np.savetxt(file, np.append(self.harmony_fit,self.WL_fit))
        with open(address+'current_harmonies.txt', 'wb') as file:
            np.savetxt(file, np.append(self.HM,self.waitingList,axis=0))
        with open(address+'computeTimes.txt', 'wb') as file:
            np.savetxt(file, np.append(self.compTimesHM,self.compTimesWL), fmt = '%d')
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
        # with open(address+'best_evolution.csv','a') as f:
        #     f.write(self.best_evolution_temp)
        with open(address+'tabuList.txt','a') as f:
            f.write(self.tabuList_temp)
        with open(address+'total_time.txt', 'w') as file:
            file.write(str(totalTime))
        with open(address+'current_cicle.txt', 'w') as file:
            file.write(str(self.iter+len(self.new_harmony)))
        self._restartSTR()

    def _restartSTR(self):
        """
        Reset the variables that hold information to be saved in the files 
        after each call to the `saveinfo()` method.

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """

        self.results = {}
        self.fitness_vs_gen_temp = ""
        self.all_harmonies_temp = ""
        self.tabuList_temp = ""

    def _new_harmony(self):
        """
        Create and update the new harmonies from the Harmony Memory (HM) using
        the NGHS operations.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.new_harmony = np.zeros((1, self.numvars))  
        itl = True
        while itl: #itl: in tabu list 
            #Create new harmony
            for j in range(self.numvars):
                if random.random() < self.pm:
                    newparam = random.uniform(self.minvalu[j],self.maxvalu[j])
                else:
                    x_r = 2 * self.HM[self.best_id, j] - self.HM[self.worst_id, j]
                    if x_r < self.minvalu[j]:
                        x_r = self.minvalu[j]
                    elif x_r > self.maxvalu[j]:
                        x_r = self.maxvalu[j]
                    newparam = self.HM[self.worst_id, j] + random.random()*(x_r-self.HM[self.worst_id, j])
                if self.param_accuracy is not None:
                    newparam = np.round(newparam, self.param_accuracy[j])
                self.new_harmony[0,j] = newparam 
            itl = np.any(np.all(self.tabuList == self.new_harmony[0], axis=1))