import sys
import psutil
import numpy as np
import ray
from scipy.optimize import linear_sum_assignment
from fastphase import calc_func

### RAY FUNCTIONS

hap_calc = ray.remote(calc_func.hapCalc)
gen_calc = ray.remote(calc_func.genCalc)
lik_calc = ray.remote(calc_func.likCalc)

hap_viterbi = ray.remote(calc_func.hapViterbi)
gen_viterbi = ray.remote(calc_func.genViterbi)

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
    rez = np.where(gen <0, np.sum(pz*theta_sum_mat,axis=(1,2)), gen)
    return rez

@ray.remote
def calc_cost_matrix_haplo( h, nK ):
    nL= h.shape[0]
    try:
        assert nL>0
    except AssertionError:
        print (*h)
    res = np.zeros( ( nL-1, nK, nK), dtype=np.float)
    for l in range(nL -1):
        res[l, h[l], h[l+1]] -= 1
    return res

@ray.remote
def calc_cost_matrix_geno( g, nK):
    nL= g.shape[0]
    res = np.zeros( ( nL-1, nK, nK), dtype=np.float)
    for l in range(nL -1):
        res[l,] = calc_cost_matrix_geno_item( g[l:(l+2),], nK )
    return res

def calc_cost_matrix_geno_item( vit, nK):
    k1, k2 = vit[0]
    kp1, kp2 = vit[1]
    cost_mat = np.zeros((nK,nK))
    if (k1 == kp1):
        if (k2 == kp2): ## k1 == kp1 & k2 == kp2 , possibly k1 == k2
            cost_mat[k1, kp1] -= 1
            cost_mat[k2, kp2] -= 1
        else: ## k1 == kp1 & k2 != kp2
            cost_mat[k1, kp1] -= 1 ## diagonal
            cost_mat[k2, kp2] -= 1
    elif ( k2 == kp2): ## k1 != kp1 & k2 == kp2
        cost_mat[k1, kp1] -= 1 ## diagonal
        cost_mat[k2, kp2] -= 1
    elif ( k1 == kp2) and (k2 != kp1):
        cost_mat[k1, kp2] -= 1 ## diagonal
        cost_mat[k2, kp1] -= 1
    elif ( k1 != kp2) and (k2 == kp1):
        cost_mat[k1, kp2] -= 1 ## diagonal
        cost_mat[k2, kp1] -= 1
    else:
        cost_mat[ k1, kp1] -= 0.5
        cost_mat[ k1, kp2] -= 0.5
        cost_mat[ k2, kp1] -= 0.5
        cost_mat[ k2, kp2] -= 0.5
    return cost_mat

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
        self.rho=self.jm/self.nhap
        self.rho = np.where(self.rho<0, self.rhomin, self.rho)
        ## alpha
        if self.alpha_up:
            self.alpha = self.jmk/self.jm
            self.alpha = np.where( self.alpha>0.999,0.999,self.alpha)
            self.alpha = np.where( self.alpha<0.001,0.001,self.alpha)
            self.alpha /= np.sum(self.alpha, axis=1, keepdims=True)
        ## theta
        if self.theta_up:
            self.theta = self.top/self.bot
            self.theta = np.where(self.theta>0.999,0.999,self.theta)
            self.theta = np.where(self.theta<0.001,0.001,self.theta)
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
            print("Shutdown ray")
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
                del result
            ## Genotypes
            result_ids = [ gen_calc.remote(alpha, theta, rho, gen,0) for gen in self.genotypes.values()]
            while len(result_ids):
                item, result_ids = ray.wait(result_ids)
                result = ray.get(item)
                hLogLike,top,bot,jmk = result[0]
                par.addIndivFit(top,bot,jmk,2)
                log_like+=hLogLike
                del result
            ## Genotype Likelihoods
            result_ids = [ lik_calc.remote(alpha, theta, rho, lik,0) for lik in self.genolik.values()]
            while len(result_ids):
                item, result_ids = ray.wait(result_ids)
                result = ray.get(item)
                hLogLike,top,bot,jmk = result[0]
                par.addIndivFit(top,bot,jmk,2)
                log_like+=hLogLike
                del result
                
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
                del pgeno
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
                del pgeno
        return Imputations

    def viterbi( self, parList):
        Imputations = {}
        for hap in self.haplotypes:
            Imputations[hap] = []
        for gen in self.genotypes:
            Imputations[gen] = []
        for par in parList:
            alpha = ray.put(par.alpha)
            theta = ray.put(par.theta)
            rho = ray.put(par.rho)
            result_map = {}
            result_ids = []
            ## Haplotypes
            for name,hap in self.haplotypes.items():
                result_id = hap_viterbi.remote( alpha, theta, rho, hap)
                result_map[result_id] = name
                result_ids.append(result_id)
            ## Genotypes
            for name,gen in self.genotypes.items():
                result_id = gen_viterbi.remote( alpha, theta, rho, gen)
                result_map[result_id] = name
                result_ids.append(result_id)
            while len(result_ids):
                item, result_ids = ray.wait(result_ids)
                pth = ray.get(item)[0]
                name = result_map[item[0]]
                Imputations[name].append( pth)
        return Imputations
       
    def optimfit(self, nClus = 20, nstep = 10, params=None, verbose = False, rhomin=1e-6, alpha_up = False, fast=False, nEM = 5, niter=10):
        """Fit a fastphase model by optimizing likelihood via minimizing cluster switches.

        This method attempts to find an optimum (in terms of
        likelihood) set of parameters for a fastphase model, with
        equal weight clusters (by default). This is done by : 

        0. Fit nEM fastphase models on the data and keep the best one
        1. Then for niter successive iterations:
            1.2. Compute viterbi path for each data point (genotypes and
                 haplotypes). Warning : this step is in K^4 for genotype data
                 so can be extremely slow. Ok for haplotype (phased) data.
            1.3. Based on viterbi paths determine at each SNP interval if
                 cluster labels should be switched using the Hungarian method
                 (linear_sum_assignment).
            1.4. Permute cluster labels to minimize switch and iterate again
        2. At the last iteration the model is run for max(30,nstep) iterations 
        to estimate the final set of parameters.
        
        Parameters
        ----------

        nClus : int
            number of haplotype clusters
        nstep : int
            number of iteration for the EM algorithm
        params : object
            mod_params instance of starting parameters
        verbose : bool
            turn on verbosity
        rhomin : float
            minimum value for rho
        alpha_up : bool
            should the alphas be optimized or kept equal (default)
        fast : boot
            experimental, speed up computations at the expense of efficiency
        nEM : int
            number of starting EM fits
        niter : int
            number of iterations in the finale steps.
            
        Returns
        -------
        fastphase.mod_params 
        
        A set of parameters of a fastphase model.

        Note 
        ----
        Computations are parallelized across available CPUs (see fastphase.nproc)

        """
        liktraj = []

        try:
            assert(len(self.genotypes)+len(self.haplotypes) > 0)
        except:
            raise AssertionError("Optimfit only available with genotyping data")

        print("=== OPTIMFIT RUN ===",file=self.flog)
        ## Fit nEM EM to find an initial start
        if verbose:
            print("*** Init EM", 1,file=self.flog)
        nbest = 0
        curpar = self.fit( nClus = nClus, nstep = nstep, verbose = verbose, alpha_up = alpha_up, fast=fast)
        liktraj.append((0, 0, curpar.loglike))
        for n in range(1, nEM):
            if verbose:
                print("*** Init EM", n+1,file=self.flog)
            
            par = self.fit( nClus = nClus, nstep = nstep,verbose = verbose, alpha_up = alpha_up, fast=fast)
            liktraj.append((n, 0, par.loglike))
            if par.loglike > curpar.loglike:
                nbest = n
                curpar = par
        init_par = curpar
        ## Improve it
        print('Initial Viterbi',file=self.flog)
        self.flog.flush()
        imp = self.viterbi([curpar])
        
        for it in range(niter):
            if verbose:
                print("*** Iter", it+1,file=self.flog)
                self.flog.flush()
            ## calculate costs
            if verbose:
                print("Calculating Costs",file=self.flog)
                self.flog.flush()

            ## MEMORY problems here parallelize along SNPs if Matrix larger than 100 MB
            cost_mat_tot = np.zeros( (self.nLoci-1, nClus, nClus), dtype=np.float)
            result_ids = []
            i_res = 0
            for haplo in self.haplotypes.keys():
                i_res += 1
                result_id = calc_cost_matrix_haplo.remote( np.array(imp[haplo][0]), nClus)
                result_ids.append(result_id)
                if (i_res % self.nproc) == 0:
                    for res in ray.get(result_ids):
                        cost_mat_tot += res
                        del res
                    result_ids=[]
                    i_res=0
            ## terminate
            for res in ray.get(result_ids):
                cost_mat_tot += res
                del res
 
            result_ids = []
            i_res = 0
            for geno in self.genotypes.keys():
                i_res += 1
                result_id = calc_cost_matrix_geno.remote( np.array(imp[geno][0]), nClus)
                result_ids.append(result_id)
                if (i_res % self.nproc) == 0:
                    for res in ray.get(result_ids):
                        cost_mat_tot += res
                        del res
                    result_ids=[]
                    i_res=0
            ## terminate
            for res in ray.get(result_ids):
                cost_mat_tot += res
                del res
 
            
            ## mem management
            for k in list(imp.keys()):
                del imp[k]
            del imp
            
            ## combine
            if verbose:
                print("Computing optimum permutations",file=self.flog)
                self.flog.flush()
            # res = np.array( self.pool.map( linear_sum_assignment, cost_mat_tot,  1000))
            res = np.array( [ linear_sum_assignment(x) for x in cost_mat_tot ])
            ## mem management
            del cost_mat_tot
            permut = res[:,1,:]
            newpar = self.switch_pars( curpar, permut)
            if verbose:
                print("EM with switched parameters",file=self.flog)
                self.flog.flush()
            if (it == niter-1): ## last iteration, longer and no imputation
                par_s = self.fit( nClus = nClus, nstep=max(30,nstep), verbose=verbose, params=newpar, alpha_up=False, fast=fast)
            else:
                par_s = self.fit( nClus = nClus, nstep=nstep, verbose=verbose, params=newpar, alpha_up=False, fast=fast)
                imp = self.viterbi([par_s])
            curpar = par_s
            liktraj.append((nbest, it+1, curpar.loglike))
        
        ## add alpha update
        # par_s = self.fit(nClus = nClus, nstep = max(30,nstep), verbose = True, params = par_s, alpha_up = True, theta_up = False, fast = fast)
        # liktraj.append((nbest, it+2, par_s.loglike))
        print('EM','iter','loglik',file=self.flog)
        for dat in liktraj:
            print(*dat,file=self.flog)
        return par_s

    @staticmethod
    def switch_pars(par, permut):
        newpar = modParams(par.nLoc, par.nClus, alpha_up = par.alpha_up)
        for i in range(par.nClus):
            newpar.theta[0,i]=par.theta[0,i]
            curclus = i
            for j in range(1, par.nLoc):
                curclus =  permut[ j-1, curclus]
                newpar.theta[ j,i] = par.theta[ j, curclus]
        return newpar
