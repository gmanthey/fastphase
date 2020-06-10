import sys
import psutil
import numpy as np
import ray
from fastphase import calc_func

### RAY FUNCTIONS

hap_calc = ray.remote(calc_func.hapCalc)
gen_calc = ray.remote(calc_func.genCalc)
lik_calc = ray.remote(calc_func.likCalc)

@ray.remote
def hap_p_all(hap, pz, theta):
    L = hap.shape[0]
    K = pz.shape[1]
    rez = np.where( hap <0, np.einsum( 'lk->l', pz*theta), np.einsum( 'lk,l->l',pz, hap))
    return rez

@ray.remote
def gen_p_geno(gen, pz, theta):
    L = gen.shape[0]
    K = pz.shape[1]
    O = np.ones(K)
    theta_sum_mat = np.array([np.kron(O,t).reshape(K,K)+np.kron(t,O).reshape(K,K) for t in theta])
    rez = np.where(gen <0, np.einsum( 'lkk->l', pz*theta_sum_mat), np.einsum('lkk,l->l', pz, gen))
    return rez

### CLASSES
class modParams():
    '''
    A class for fastphase model parameters.
    Dimensions:
    -- number of loci: N
    -- number of clusters: K
    Parameters:
    -- theta (N x K): allele frequencies in clusters at each locus
    -- alpha (N x K): cluster weights at each locus
    -- rho (N x 1): jump probabilities in each interval
    '''
    def __init__(self,nLoc,nClus,rhomin=1e-6, alpha_up = True, theta_up = True):
        self.nLoc=nLoc
        self.nClus=nClus
        self.theta=0.98*np.random.random((nLoc,nClus))+0.01 # avoid bounds 0 and 1
        self.rho=np.ones((nLoc,1))/1000
        ##self.alpha=np.random.mtrand.dirichlet(np.ones(nClus),nLoc) # make sure sum(alpha_is)=1
        self.alpha=1.0/nClus*np.ones((nLoc,nClus))
        self.rhomin=rhomin
        self.alpha_up = alpha_up
        self.theta_up = theta_up
        self.loglike = 0
    def initUpdate(self):
        self.top=np.zeros((self.nLoc,self.nClus))
        self.bot=np.zeros((self.nLoc,self.nClus))
        self.jmk=np.zeros((self.nLoc,self.nClus))
        self.jm=np.zeros((self.nLoc,1))
        self.nhap=0.0
    def addIndivFit(self,t,b,j,nhap):
        self.top += t
        self.bot += b
        self.jmk += j
        self.jm  += np.reshape(np.sum(j,axis=1),(self.nLoc,1))
        self.nhap+=nhap
    def update(self):
        ''' Update parameters using top,bot,jmk jm probabilities'''
        ## rho
        ##self.rho=self.jm/self.nhap
        for i in range(self.nLoc):
            self.rho[i,0]=self.jm[i,0]/self.nhap
            if self.rho[i,0]<self.rhomin:
                self.rho[i,0]=self.rhomin
            elif self.rho[i,0]>(1-self.rhomin):
                self.rho[i,0]=1-self.rhomin
        ## alpha
        #self.alpha=self.jmk/self.jm
        if self.alpha_up:
            for i in range(self.nLoc):
                for j in range(self.nClus):
                    self.alpha[i,j]=self.jmk[i,j]/self.jm[i,0]
                    if self.alpha[i,j]>=0.999:
                        self.alpha[i,j]=0.999
                    elif self.alpha[i,j]<0.001:
                        self.alpha[i,j]=0.001
                self.alpha[i,:] /= np.sum(self.alpha[i,:])
        ## theta
        if self.theta_up:
            self.theta=self.top/self.bot
            for i in range(self.nLoc):
                for j in range(self.nClus):
                    if self.theta[i,j]>0.999:
                        self.theta[i,j]=0.999
                    elif self.theta[i,j]<0.001:
                        self.theta[i,j]=0.001
    def write(self,stream=sys.stdout):
        print("snp", *["t"+str(i) for i in range(self.nClus)], "rho", *["a"+str(i) for i in range(self.nClus)], file=stream)
        for i in range(self.nLoc):
            print(i, *[np.round( self.theta[i,k], 3) for k in range(self.nClus)], np.round( self.rho[i,0], 7), *[np.round( self.alpha[i,k], 3) for k in range(self.nClus)], file=stream)


class fastphase():
    '''
    A class to manipulate and control a fastphase model (Scheet and Stephens, 2006)
    Initialized with a problem size = number of loci

    Usage : 
    with fastphase(nloc, nproc) as fph:
         ... do stuff ...
    '''
    def __init__(self, nLoci, nproc = psutil.cpu_count(),prfx=None):
        assert nLoci>0
        self.nLoci=nLoci
        self.haplotypes={}
        self.genotypes={}
        self.genolik={}
        self.nproc = nproc
        self.prfx = prfx
        self.init_ray=False ## did we initialized ray

    def __enter__(self):
        if not ray.is_initialized():
            print("Initializing ray")
            ray.init(num_cpus=self.nproc)
            print(ray.nodes())
            self.init_ray=True
        else:
            print("Using already initialized ray")
        if self.prfx is None:
            self.flog = sys.stdout
        else:
            self.flog = open(self.prfx, 'w')
        return self

    def __exit__(self,*args):
        if self.init_ray:
            print("Killing ray")
            ray.shutdown()
        if self.prfx is not None:
            self.flog.close()

    def flush(self):
        '''
        remove data 
        '''
        for v in self.haplotypes.values():
            del v
        for v in self.genotypes.values():
            del v
        for v in self.genolik.values():
            del v
        self.haplotypes={}
        self.genotypes={}
        self.genolik={}
    
    def addHaplotype(self,ID,hap,missing=-1):
        '''
        Add an haplotype to the model observations.
        hap is a numpy array of shape (1,nLoci).
        Values must be 0,1 or missing
        '''
        try:
            assert hap.shape[0]==self.nLoci
            self.haplotypes[ID]=ray.put(hap)
        except AssertionError:
            print("Wrong Haplotype Size:",hap.shape[0],"is not",self.nLoci)
            raise
    def addGenotype(self,ID,gen,missing=-1):
        '''
        Add a genotype to the model observations.
        gen is a numpy array of shape (1,nLoci).
        Values must be 0,1,2 or missing
        '''
        try:
            assert gen.shape[0]==self.nLoci
            self.genotypes[ID]=ray.put(gen)
        except AssertionError:
            print("Wrong Genotype Size:",gen.shape[0],"is not",self.nLoci)
            raise
    def addGenotypeLikelihood(self, ID, lik):
        '''
        Add a matrix of genotype likelihoods to the model observations.
        lik is a numpy array of shape (nLoci,3).
        Values are (natural)log-likelihoods log( P( Data | G=0,1,2) )
        '''
        try:
            assert lik.shape == (self.nLoci,3)
        except AssertionError:
            print("Wrong Array Size:", lik.shape,"is not",(self.nLoci,3))
            raise
        ##lik = np.array(lik) 
        self.genolik[ID] = ray.put(lik - np.max(lik, axis=1,keepdims=True))
            
    @staticmethod
    def gen2hap(gen):
        return np.array( _tohap(np.array(gen, dtype=int)), dtype=int)
    
    def fit(self,nClus=20,nstep=20,params=None,verbose=False,rhomin=1e-6, alpha_up = True, theta_up = True, fast=False):
        '''
        Fit the model on observations with nCLus clusters using nstep EM iterations
        Multithread version using ray.
        '''
        try:
            assert ray.is_initialized()
        except AssertionError:
            print('Usage :\n\t with fastphase(nloc, nproc) as fph: \n ...')
            raise
            
        if params:
            par=params
            par.alpha_up = alpha_up
            par.theta_up = theta_up
        else:
            par=modParams(self.nLoci,nClus,rhomin, alpha_up, theta_up)
        if verbose:
            print( 'Fitting fastphase model',file=self.flog)
            print( '# clusters ',nClus, file=self.flog)
            print( '# threads ', self.nproc, file=self.flog)
            print( '# Loci', self.nLoci, file=self.flog)
            print( '# Haplotypes',len(self.haplotypes), file=self.flog)
            print( '# Genotypes', len(self.genotypes), file=self.flog)
            print( '# Likelihoods', len(self.genolik), file=self.flog)
        old_log_like=1

        for iEM in range(nstep):
   
            log_like=0.0
            par.initUpdate()
            alpha = ray.put(par.alpha)
            theta = ray.put(par.theta)
            rho = ray.put(par.rho)

            ## Haplotypes
            result_ids = [ hap_calc.remote(alpha, theta, rho, hap,0) for hap in self.haplotypes.values()]
            while len(result_ids):
                item, result_ids = ray.wait(result_ids)
                result = ray.get(item)
                hLogLike,top,bot,jmk = result[0]
                par.addIndivFit(top,bot,jmk,1)
                log_like+=hLogLike
            ## Genotypes
            result_ids = [ gen_calc.remote(alpha, theta, rho, gen,0) for gen in self.genotypes.values()]
            while len(result_ids):
                item, result_ids = ray.wait(result_ids)
                result = ray.get(item)
                hLogLike,top,bot,jmk = result[0]
                par.addIndivFit(top,bot,jmk,2)
                log_like+=hLogLike
            ## Genotype Likelihoods
            result_ids = [ lik_calc.remote(alpha, theta, rho, lik,0) for lik in self.genolik.values()]
            while len(result_ids):
                item, result_ids = ray.wait(result_ids)
                result = ray.get(item)
                hLogLike,top,bot,jmk = result[0]
                par.addIndivFit(top,bot,jmk,2)
                log_like+=hLogLike
                
            ## remove parameters from ray object store
            del alpha
            del theta
            del rho
            if verbose:
                print( iEM, log_like, file=self.flog)
                self.flog.flush()
            par.update()
            par.loglike=log_like
        
        return par

    def impute(self,parList):
        Imputations = {}
        for hap in self.haplotypes:
            Imputations[hap] = [ np.zeros( self.nLoci, dtype=np.float), []] ## P_geno, probZ, Path
        for gen in self.genotypes:
            Imputations[gen] = [ np.zeros( self.nLoci, dtype=np.float), []] ## P_geno, probZ, Path
        for lik in self.genolik:
            Imputations[lik] = [ np.zeros( (self.nLoci,3), dtype=np.float), []] ## P_geno, probZ, Path
        x = 1.0/len(parList)
        
        for par in parList:
            alpha = ray.put(par.alpha)
            theta = ray.put(par.theta)
            rho = ray.put(par.rho)
            ## Haplotypes
            result_map = {}
            pz_map = {} ## maps name -> P(Z|G)
            result_ids = []
            for name,hap in self.haplotypes.items():
                pz_id = hap_calc.remote(alpha, theta, rho, hap, 1)
                pz_map[name]=pz_id
                result_id = hap_p_all.remote(hap, pz_id, theta)
                result_map[result_id] = name
                result_ids.append(result_id)
            while len(result_ids):
                item, result_ids = ray.wait(result_ids)
                pall = ray.get(item)[0]
                name = result_map[item[0]]
                Imputations[name][0] += x*pall
                Imputations[name][1].append(ray.get(pz_map[name]))
            ## Genotypes
            result_map = {} ## maps result_id (P(G|theta)) -> name
            pz_map = {} ## maps name -> P(Z|G)
            result_ids = []
            for name,gen in self.genotypes.items():
                pz_id = gen_calc.remote(alpha, theta, rho, gen, 1)
                pz_map[name]=pz_id
                result_id = gen_p_geno.remote(gen, pz_id, theta)
                result_map[result_id] = name
                result_ids.append(result_id)
            while len(result_ids):
                item, result_ids = ray.wait(result_ids)
                pgeno = ray.get(item)[0]
                name = result_map[item[0]]
                Imputations[name][0] += x*pgeno
                Imputations[name][1].append(ray.get(pz_map[name]))
            ## Genotype Likelihoods
            result_map = {} ## maps result_id (P(G|theta)) -> name
            result_ids = []
            for name,lik in self.genolik.items():
                result_id = lik_calc.remote(alpha, theta, rho, lik, 1)
                result_map[result_id] = name
                result_ids.append(result_id)
            while len(result_ids):
                item, result_ids = ray.wait(result_ids)
                pZ,pgeno = ray.get(item)[0]
                name = result_map[item[0]]
                Imputations[name][0] += x*pgeno
                Imputations[name][1].append(pZ)

        return Imputations
