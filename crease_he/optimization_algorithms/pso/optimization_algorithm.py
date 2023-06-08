import numpy as np
import random
import math

class optimization_algorithm:
    """
    """

    def __init__(self,
                 optim_params = [5, 10],
                 adapt_params = [0.9,0.2,2,2]):
        self._name = "pso"
        self._numadaptparams = 2
        self._numoptimparams = 3
        self.pop_number = optim_params[0]
        self.generations = optim_params[1]

        self.wMax = adapt_params[0]
        self.wMin = adapt_params[1]
        self.c1 = adapt_params[2]
        self.c2 = adapt_params[3]
        self.bestfit = np.inf

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
        self.minvalu = np.array(minvalu)
        self.maxvalu = np.array(maxvalu)
        self.numvars = len(minvalu)
        self.deltavalu = self.maxvalu-self.minvalu
        self.vMax = np.subtract(maxvalu,minvalu)*0.2
        self.vMin = -self.vMax
    
    def update_pop(self, fit, generation):
        np.savetxt(self.address+'population_'+str(generation)+'.txt',np.c_[self.pop])
        #Save the individuals of the generation i in file results_i.txt
        F1= open(self.address+'results_'+str(generation)+'.txt','w')
        F1.write('#individual...all params...error\n')

        for val in range(self.pop_number): 
            #Save the params ofthe individual val
            F1.write(str(val)+' ')
            for p in self.pop[val]:
                F1.write(str(p)+' ')
            F1.write(str(fit[val])+'\n')
            F1.flush()
        ### returns cummulative relative error from which individuals can be selected ###
        maxfit=np.min(fit)
        improved = (maxfit <= self.bestfit)
        elitei=np.where(fit==maxfit)[0]                  # Best candidate 
        secondfit=sorted(fit)[1]
        secondi = np.where(fit==secondfit)[0]            # Second best candidate
        avgfit=np.average(fit)
        avgi=np.array([(np.abs(fit-avgfit)).argmin()])   # Average candidate
        minfit=np.max(fit)
        mini=np.where(fit==minfit)[0]                    # Worst candidate

        if len(elitei)>1:
            elitei=elitei[0]
        if len(secondi)>1:
            secondi=secondi[0]
        if len(avgi)>1:
            avgi=avgi[0]
        if len(mini)>1:
            mini=mini[0]
        
        f = open(self.address+'fitness_vs_gen.txt', 'a' )
        if generation == 0:
            f.write( 'gen mini min avgi avg secondi second besti best\n' )
        f.write( '%d ' %(generation) )
        f.write( '%d %.8lf ' %(mini,minfit) )
        f.write( '%d %.8lf ' %(avgi,avgfit) )
        f.write( '%d %.8lf ' %(secondi,secondfit) )
        f.write( '%d %.8lf ' %(elitei,maxfit) )           #
        f.write( '\n' )
        f.close()
        print('Generation best fitness: {:.4f}'.format(maxfit))
        print('Generation best parameters '+str(self.pop[elitei]))
        #IQid_str = np.array(IQid_str) #TODO Fix it
        #with open(self.address+'IQid_best.txt','a') as f:
        #    f.write(np.array2string(IQid_str[elitei][0])+'\n')

        w=self.wMax-generation*((self.wMax-self.wMin)/self.generations)

        for k in range(self.pop_number):
            currentX=self.pop[k,:]
            fitness=fit[k]
            #update PBEST
            if fitness<self.PBEST_O[k]:
                self.PBEST_X[k,:]=currentX
                self.PBEST_O[k]=fitness
            #update GBEST
            if fitness<self.GBEST_O:
                self.GBEST_X=currentX
                self.GBEST_O=fitness
        for k in range(self.pop_number):
            self.V[k,:]=w*self.V[k,:]+self.c1*np.multiply(np.random.rand(self.numvars),np.subtract(self.PBEST_X[k,:],self.pop[k,:]))+self.c2*np.multiply(np.random.rand(self.numvars),np.subtract(self.GBEST_X,self.pop[k,:]))
            index1 = np.argwhere(self.V[k, :] > self.vMax)
            index2 = np.argwhere(self.V[k, :] < self.vMin)
            self.V[k, index1] = self.vMax[index1]
            self.V[k, index2] = self.vMin[index2]
            self.pop[k,:]=self.pop[k,:]+self.V[k,:]
            index1=np.argwhere(self.pop[k,:]>self.maxvalu)
            index2=np.argwhere(self.pop[k,:]<self.minvalu)
            self.pop[k,index1]=self.maxvalu[index1]
            self.pop[k,index2]=self.minvalu[index2]

        ### save output from current generation in case want to restart run
        np.savetxt(self.address+'current_cicle.txt',np.c_[generation+1])
        np.savetxt(self.address+'current_pop.txt',np.c_[self.pop])

        #'''
        np.savetxt(self.address + 'current_V.txt', np.c_[self.V])
        np.savetxt(self.address + 'current_PBEST_O.txt', np.c_[self.PBEST_O])
        np.savetxt(self.address + 'current_PBEST_X.txt', np.c_[self.PBEST_X])
        np.savetxt(self.address + 'current_GBEST_O.txt', np.c_[self.GBEST_O])
        np.savetxt(self.address + 'current_GBEST_X.txt', np.c_[self.GBEST_X])

        
        #'''

        print(self.GBEST_O)

        return self.pop, improved
        

    def resume_job(self, address):
        self.address = address
        self.pop = np.zeros((self.pop_number, self.numvars))
        self.pop = np.genfromtxt(self.address+'current_pop.txt') #
        generation = int(np.genfromtxt(self.address+'current_cicle.txt'))
        #temp = np.genfromtxt(self.address+'current_w.txt')
        #self.W = temp
        #'''
        self.PBEST_O = np.genfromtxt(self.address+'current_PBEST_O.txt')
        self.PBEST_X = np.genfromtxt(self.address + 'current_PBEST_X.txt')
        self.GBEST_O = np.genfromtxt(self.address + 'current_GBEST_O.txt')
        self.GBEST_X = np.genfromtxt(self.address + 'current_GBEST_X.txt')
        self.V = np.genfromtxt(self.address + 'current_V.txt')
        #'''
        print('Restarting from generation #{:d}'.format(generation))
        return generation, self.pop
    
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
        self.pop = np.zeros((self.pop_number,self.numvars))
        self.V=np.zeros((self.pop_number,self.numvars))
        self.PBEST_X=np.zeros((self.pop_number,self.numvars))
        self.GBEST_X=np.zeros((self.pop_number,self.numvars))
        self.PBEST_O=np.zeros(self.pop_number)
        self.GBEST_O=math.inf

        for k in range(self.pop_number):
            self.pop[k,:]=np.add(np.multiply(np.subtract(self.maxvalu,self.minvalu),np.random.rand(self.numvars)),self.minvalu)
            self.V[k,:]=np.zeros(self.numvars)
            self.PBEST_X[k,:]=np.zeros(self.numvars)
            self.PBEST_O[k]=math.inf
            self.GBEST_X=np.zeros(self.numvars)

        print('New run')
        return self.pop