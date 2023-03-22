import numpy as np
import random

class optimization_algorithm:
    """
    """

    def __init__(self,
                 optim_params = [5,10,7],
                 adapt_params = [0.005,0.85,0.1,1,0.006,0.25,1.1,0.6,0.001]):
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
        self._numparams = 3
        self._name = "ga"

    @property
    def numparams(self):
        return self._numparams
    
    @property
    def name(self):
        return self._name
    
    def genetic_operations(self,pop,pacc,elitei):
        popn = np.zeros(np.shape(pop))
        cross = 0
        mute = 0
        pc = self.adaptation_params.pc
        pm = self.adaptation_params.pm
        
        for i in range(self.popnumber-1):
            #####################    Crossover    ####################
            #Selection based on fitness
            testoff=random.random()
            isit=0
            npart1=1
            for j in range(1,self.popnumber):
                if (testoff>pacc[j-1])&(testoff<pacc[j]):
                    npart1=j

            testoff=random.random()
            isit=0
            npart2=1
            for j in range(self.popnumber):
                if (testoff>=pacc[j-1])&(testoff!=pacc[j]):
                    npart2=j

            #Fit parents put in array popn
            popn[i,:]=pop[npart1,:]

            testoff=random.random()
            loc=int((testoff*(self.numvars-1))*self.nloci)
            if loc==0:
                loc=self.nloci
            testoff=random.random()

            #crossover
            if (testoff<=pc):
                cross+=1
                popn[i,loc:]=pop[npart2,loc:]


        #####################    Mutation    ####################
            for j in range(self.nloci*self.numvars):
                testoff=random.random()
                if (testoff<=pm):
                    popn[i,j]=random.randint(0,1)
                    mute+=1

        #####################    Elitism    ####################
        popn[-1,:]=pop[elitei,:]

        
        print('pc',pc)
        print('#crossovers',cross)
        print('pm',pm)
        print('#mutations',mute)
        print('\n')
        
        return popn

    def update(self,gdm):
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

    def resume_job(self, address):
        currentcicle = int(np.genfromtxt(address+'current_cicle.txt'))
        self.pop = np.genfromtxt(address+'current_pop.txt')
        temp = np.genfromtxt(address+'current_pm_pc.txt')
        self.pm = temp[0]
        self.pc = temp[1]
        # read in best iq for each generation
        bestIQ = np.genfromtxt(address+'best_iq.txt')
        # do not include q values in bestIQ array
        bestIQ = bestIQ[1:,:]
        print('Restarting from gen #{:d}'.format(currentcicle+1))
        return currentcicle, self.pop
    
    def new_job(self, numvars):
        self.pop = initial_pop(self.pop_number, self.nloci, numvars)
        return self.pop
    
def initial_pop(popnumber, nloci, numvars):
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
    pop=np.zeros((popnumber,nloci*numvars))
    for i in range(popnumber):
        for j in range(nloci*numvars):
            randbinary=np.random.randint(2)
            pop[i,j]=randbinary
    return pop