import numpy as np
import random

class optimization_algorithm:
    """
    Class for the GA (Genetic Algorithm of Dinamyc Adaptation).

    Attributes
    ----------
    hyperparameters : list of length 3, default=[5, 10, 7]
        A list containing the hyperparameters required for the GA.
        The list corresponds to [`pop_number`, `generations`, `nloci`].
        - pop_number : int, default = 5
            Population size.
        - generations : int, default = 10
            Number of generations.
        - nloci : int, default = 7
            Numbers of bits to represente each Shape parameter.

    adapt_hyperparams : list of length 9, default=[0.005,0.85,0.1,1,0.006,0.25,1.1,0.6,0.001]
        Additional hyperparameters required for the execution of the optimization 
        algorithm. These parameters are not used in the NGHS algorithm. The list
        corresponds to [`gdmmin`, `gdmmax`, `pcmin`, `pcmax`, `pmmin`, `pmmax`,
        `kgdm`, `pc`, `pm`].
        gdmmin: float. Default=0.005.
            The minimum acceptable value of gdm, a measurement of diversity within
            a generation (high gdm means low diversity, and vice versa). If gdm of
            the current generation falls below `gdmmin`, `pc` will be multiplied by
            `kgdm` and `pm` will be divided by `kgdm` to reduce diversity.
        gdmmax: float. Default=0.85.
            The maximum acceptable value of gdm, a measurement of diversity within
            a generation (high gdm means low diversity, and vice versa). If gdm of
            the current generation exceeds `gdmmax`, `pc` will be divided by
            `kgdm` and `pm` will be multiplied by `kgdm` to increase diversity.
        pcmin: float. Default=0.1.
            Minimum value of `pc`. `pc` cannot be further adjusted below `pcmin`,
            even if `gdm` is still too high.
        pcmax: float. Default=1.
            Maximum value of `pc`. `pc` cannot be further adjusted above `pcmax`,
            even if `gdm` is still too low.
        pmmin: float. Default=0.006.
            Minimum value of `pm`. `pm` cannot be further adjusted below `pmmin`,
            even if `gdm` is still too low.
        pmmax: float. Default=0.25.
            Maximum value of `pm`. `pm` cannot be further adjusted above `pmmax`,
            even if `gdm` is still too high.
        kgdm: float. Default=1.1.
            Should be > 1. The magnitude of adjustment for `pc` and `pm` in case
            `gdm`
            falls ouside of [ `gdmmin`,`gdmmax` ].
        pc: float. Default=0.6.
            Inicial Possibility of a crossover action happening on an individual
            in the next generation. `pc` is updated after each generation according
            to `gdm`.
        pm: float. Default=0.001.
            Inicial possibillity of a mutation action happening on each gene in an
            individual. `pm` is updated after each generation according to `gdm`.

    param_accuracy : list, default=None
        This Atributte is not used in the GA.

    waitinglistSize : int, default=None
        This Atributte is not used in the GA.

    maxComputeTime : int, default=None
        This Atributte is not used in the GA.
    """

    def __init__(self,
                 hyperparameters = [5, 10, 7],
                 adapt_hyperparams = [0.005,0.85,0.1,1,0.006,0.25,1.1,0.6,0.001],
                 param_accuracy = None,
                 waitinglistSize = None,
                 maxComputeTime = None):
        self._name = "ga"
        self._numadaptparams = 9
        self._numoptimparams = 3
        self.pop_number = hyperparameters[0]
        self.generations = hyperparameters[1]
        self.nloci = hyperparameters[2]
        self.gdmmin = adapt_hyperparams[0]
        self.gdmmax = adapt_hyperparams[1]
        self.pcmin = adapt_hyperparams[2]
        self.pcmax = adapt_hyperparams[3]
        self.pmmin = adapt_hyperparams[4]
        self.pmmax = adapt_hyperparams[5]
        self.kgdm = adapt_hyperparams[6]
        self.pc = adapt_hyperparams[7]
        self.pm = adapt_hyperparams[8]
        self.bestfit = np.inf
        self.seed = None

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
        self.deltavalu = self.maxvalu-self.minvalu
    
    def update_pop(self, fit, generation, tic, Tic):
        """
        Update the population (pop) with the results of the current generation.

        Parameters
        ----------
        fit : list
            A list containing the fitness values of the current population.
        generation : int
            The current generation number.
        tic : list
            A list containing the computation time (machine time) taken 
            to evaluate each solution in the current generation.
        Tic : float
            The total computation time for the entire genration.

        Returns
        -------
        pop : ndarray of shape (`pop_number`, `nloci`*`numvars`)
            The new population to be evaluated in the next generation, in its
            binary representation.
        improved : int
            The index of the best solution present in the current population.
        """
        #np.savetxt(self.address+'population_'+str(generation)+'.txt',np.c_[self.pop_disc])
        popn = np.zeros(np.shape(self.pop_disc))
        cross = 0
        mute = 0
        pc = self.pc
        pm = self.pm

        cs=10
        maxerr=np.max(fit)           #determines maximum SSerror for the population
        fitn=np.zeros(self.pop_number)
        fitn=np.subtract(maxerr,fit) #determines error differences
        bestfit=np.max(fitn)
        sumup=np.sum(fitn)

        avgfit=np.true_divide(sumup,self.pop_number)
        dval=bestfit-avgfit
        ascale=np.true_divide(avgfit,dval)*(cs-1.0)     #linear scaling with cs as a scaleFactor
        bscale=avgfit*(1.0-ascale)

        fitnfr=np.zeros(self.pop_number)
        
        #Save the individuals of the generation i in file results_i.txt
        F1= open(self.address+'results_'+str(generation)+'.txt','w')
        F1.write('#individual...all params...time...error\n')

        for val in range(self.pop_number): 
            #Save the params ofthe individual val
            F1.write(str(val)+' ')
            for p in self.pop[val]:
                F1.write(str(p)+' ')
            F1.write(str(tic[val])+' ')
            F1.write(str(fit[val])+'\n')
            F1.flush()
            # get scaled fitness to enable selection of bad candidates
            if (fitn[val]>avgfit):
                fitnfr[val]=ascale*fitn[val]+bscale
            else:
                fitnfr[val]=fitn[val]

        sumup=np.sum(fitnfr)

        pacc=np.zeros(self.pop_number)
        prob=np.true_divide(fitnfr,sumup)
        pacc=np.cumsum(prob)

        ### returns cummulative relative error from which individuals can be selected ###
        maxfit=np.min(fit)
        elitei=np.where(fit==maxfit)[0]                  # Best candidate 
        secondfit=sorted(fit)[1]
        secondi = np.where(fit==secondfit)[0]            # Second best candidate
        avgfit=np.average(fit)
        avgi=np.array([(np.abs(fit-avgfit)).argmin()])   # Average candidate
        minfit=np.max(fit)
        mini=np.where(fit==minfit)[0]                    # Worst candidate
        if avgfit==0:
            avgfit=1
        gdm=np.true_divide(maxfit,avgfit)
        elitei=elitei[0]
        improved = elitei
        if len(secondi)>1:
            secondi=secondi[0]
        if len(avgi)>1:
            avgi=avgi[0]
        if len(mini)>1:
            mini=mini[0]
        
        f = open(self.address+'currentState/fitness_vs_gen.txt', 'a' )
        if generation == 0:
            f.write( 'gen mini min avgi avg secondi second besti time best\n' )
        f.write( '%d ' %(generation) )
        f.write( '%.3lf %.3lf ' %(np.sum(tic), Tic) )
        f.write( '%d %.8lf ' %(mini,minfit) )
        f.write( '%d %.8lf ' %(avgi,avgfit) )
        f.write( '%d %.8lf ' %(secondi,secondfit) )
        f.write( '%d %.3lf %.8lf ' %(elitei,tic[elitei],maxfit) )
        f.write( '\n' )
        f.close()
        print('Generation best fitness: {:.4f}'.format(maxfit))
        print('Generation gdm: {:.3f}'.format(gdm))
        print('Generation best parameters '+str(self.pop[elitei]))
        #IQid_str = np.array(IQid_str) #TODO Fix it
        #with open(self.address+'IQid_best.txt','a') as f:
        #    f.write(np.array2string(IQid_str[elitei][0])+'\n')

        for i in range(self.pop_number-1):
            #####################    Crossover    ####################
            #Selection based on fitness
            testoff=random.random()
            isit=0
            npart1=1
            for j in range(1,self.pop_number):
                if (testoff>pacc[j-1])&(testoff<pacc[j]):
                    npart1=j

            testoff=random.random()
            isit=0
            npart2=1
            for j in range(self.pop_number):
                if (testoff>=pacc[j-1])&(testoff!=pacc[j]):
                    npart2=j

            #Fit parents put in array popn
            popn[i,:]=self.pop_disc[npart1,:]

            testoff=random.random()
            loc=int((testoff*(self.numvars-1))*self.nloci)
            if loc==0:
                loc=self.nloci
            testoff=random.random()

            #crossover
            if (testoff<=pc):
                cross+=1
                popn[i,loc:]=self.pop_disc[npart2,loc:]


        #####################    Mutation    ####################
            for j in range(self.nloci*self.numvars):
                testoff=random.random()
                if (testoff<=pm):
                    popn[i,j]=random.randint(0,1)
                    mute+=1

        #####################    Elitism    ####################
        popn[-1,:]=self.pop_disc[elitei,:]

        self.pop_disc = popn    
        
        self.decode()
        
        print('pc',pc)
        print('#crossovers',cross)
        print('pm',pm)
        print('#mutations',mute)
        
        self.update_adapt_params(gdm)
        ### save output from current generation in case want to restart run
        np.savetxt(self.address+'currentState/current_cicle.txt',np.c_[generation+1])
        np.savetxt(self.address+'currentState/current_pop.txt',np.c_[self.pop_disc])
        np.savetxt(self.address+'currentState/current_pm_pc.txt',np.c_[self.pm,self.pc])

        return self.pop, improved

    def update_adapt_params(self, gdm):
        '''
        Update `pc` and `pm` according to a gdm value.

        Parameters
        ----------
        gdm : float
            Current gdm value.
        
        Returns
        -------
        None
        '''
        if (gdm > self.gdmmax):
            self.pm *= self.kgdm
            self.pc = np.true_divide(self.pc,self.kgdm)
        elif (gdm < self.gdmmin):
            self.pm = np.true_divide(self.pm,self.kgdm)
            self.pc *= self.kgdm
        if (self.pm > self.pmmax):
            self.pm = self.pmmax
        if (self.pm < self.pmmin):
            self.pm = self.pmmin
        if (self.pc > self.pcmax):
            self.pc = self.pcmax
        if (self.pc < self.pcmin):
            self.pc = self.pcmin

    def resume_job(self, address, deltaiter):
        """
        Resume execution from the files located in the specified folder.

        Parameters
        ----------
        address : str
            The relative path to the folder containing the execution information.
        deltainter : int
            This Parameter isn't used in the GA.

        Returns
        -------
        generation : int
            The last genration of the execution saved in the files.
        pop : ndarray of shape (`pop_number`, `nloci`*`numvars`)
            The current new population to be evaluated, in its binary codification.
        Tic : float
            The time in seconds that the execution has been running.
        """
        self.address = address
        self.pop_disc = np.genfromtxt(self.address+'currentState/current_pop.txt')
        self.pop = np.zeros((self.pop_number,self.numvars))
        generation = int(np.genfromtxt(self.address+'currentState/current_cicle.txt'))
        temp = np.genfromtxt(self.address+'currentState/current_pm_pc.txt')
        total_time = np.genfromtxt(self.address+'currentState/total_time.txt')
        self.pm = temp[0]
        self.pc = temp[1]
        self.decode()
        print('Restarting from generation #{:d}'.format(generation))
        return generation, self.pop, total_time
    
    def new_job(self, address):
        '''
        Starts a new execution by initializing the population (pop) and other 
        necessary variables, and creates the initial files containing information 
        about the execution.

        Parameters
        ----------
        address : str
            The relative path to the folder where the execution information will be
            saved.
        
        Returns
        -------
        pop: ndarray of size (`pop_number`,`nloci`*numvars`)
            A numpy array of binary bits representing the entire first generation, 
            with each row representing a chromosome.
        '''
        self.address = address
        
        with open(address+'info.txt', 'a' ) as fi:
            fi.write( '\nPop size: %d' %(self.pop_number) )
            fi.write( '\nTotalGens: %d' %(self.generations) )
            fi.write( '\nnloci: %d' %(self.nloci) )
            fi.write( '\nGDMmin: %.4lf' %(self.gdmmin) )
            fi.write( '\nGDMmax: %.4lf' %(self.gdmmax) )
            fi.write( '\nPCmin: %.4lf' %(self.pcmin) )
            fi.write( '\nPCmax: %.4lf' %(self.pcmax) )
            fi.write( '\nPCo: %.4lf' %(self.pc) )
            fi.write( '\nPMmin: %.4lf' %(self.pmmin) )
            fi.write( '\nPMmax: %.4lf' %(self.pmmax) )
            fi.write( '\nPMo: %.4lf' %(self.pm) )
            fi.write( '\nkGDM: %.4lf' %(self.kgdm) )

        self.pop_disc = np.zeros((self.pop_number,self.nloci*self.numvars))
        self.pop = np.zeros((self.pop_number,self.numvars))
        for i in range(self.pop_number):
            for j in range(self.nloci*self.numvars):
                randbinary=random.randint(0,1)
                self.pop_disc[i,j]=randbinary
        self.decode()
        print('New run')
        return self.pop
    
    def decode(self):
        '''
        Convert a binary chromosome from a generation back to decimal Shape parameter values.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        #   decodes from binary to values between max and min
        for k in range(self.pop_number):
            valdec=np.zeros(self.numvars)
            for j in range(self.numvars): 
                n=self.nloci
                for i in range(j*self.nloci,(j+1)*self.nloci):
                    n=n-1
                    valdec[j]+=self.pop_disc[k,i]*(2**n)        
                self.pop[k][j]=self.minvalu[j]+np.true_divide((self.deltavalu[j])*(valdec[j]),2**self.nloci)

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
        address = self.address+"/currentState/"
        with open(address+'all_iq.txt', 'a') as f:
            np.savetxt(f,allIQ)
        if bestIQ is not None:
            with open(address+'best_iq.txt', 'a') as f:
                np.savetxt(f,bestIQ)
        with open(address+'total_time.txt', 'w') as file:
            file.write(str(totalTime))

    