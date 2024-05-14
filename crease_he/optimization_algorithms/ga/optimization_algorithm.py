import numpy as np
import random

class optimization_algorithm:
    """
    """

    def __init__(self,
                 optim_params = [5, 10, 7],
                 adapt_params = [0.005,0.85,0.1,1,0.006,0.25,1.1,0.6,0.001]):
        self._name = "ga"
        self._numadaptparams = 9
        self._numoptimparams = 3
        self.pop_number = optim_params[0]
        self.generations = optim_params[1]
        self.nloci = optim_params[2]
        self.gdmmin = adapt_params[0]
        self.gdmmax = adapt_params[1]
        self.pcmin = adapt_params[2]
        self.pcmax = adapt_params[3]
        self.pmmin = adapt_params[4]
        self.pmmax = adapt_params[5]
        self.kgdm = adapt_params[6]
        self.pc = adapt_params[7]
        self.pm = adapt_params[8]
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
        self.deltavalu = self.maxvalu-self.minvalu
    
    def update_pop(self, fit, generation, tic, Tic):

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
        Convert a binary chromosome from a generation back to decimal parameter values.

        Parameters
        ----------
        pop: np.array.
            A numpy array of binary bits representing the entire generation, 
            with each row representing a chromosome.
        indiv: int.
            The row ID of the chromosome of interest.
        nloci: int
            Number of binary bits used to represent each parameter in a chromosome.
        minvalu, maxvalu: list-like.
            The minimum/maximum boundaries (in decimal value) of each parameter.
            "All-0s" in binary form will be converted to the minimum for a
            parameter, and "all-1s" will be converted to the maximum.

        Returns
        -------
        param: np.array.
            A 1D array containing the decimal values of the input parameters.
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
        address = self.address+"/currentState/"
        with open(address+'all_iq.txt', 'a') as f:
            np.savetxt(f,allIQ)
        if bestIQ is not None:
            with open(address+'best_iq.txt', 'a') as f:
                np.savetxt(f,bestIQ)
        with open(address+'total_time.txt', 'w') as file:
            file.write(str(totalTime))

    