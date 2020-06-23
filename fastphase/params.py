import sys
import numpy as np

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
    @classmethod
    def from_dict(cls, d):
        try:
            par = cls(d['nLoc'], d['nClus'])
        except KeyError:
            print("Dictionary must have 'nLoc' and 'nClus' keys")
            raise
        if 'rhomin' in d:
            par.rhomin=d['rhomin']
        if 'alpha_up' in d:
            par.alpha_up = d['alpha_up']
        if 'theta_up' in d:
            par.theta_up = d['theta_up']
        if 'alpha' in d:
            par.alpha = d['alpha']
        if 'theta' in d:
            par.theta = d['theta']
        if 'rho' in d:
            par.rho = d['rho']
        return par
    
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
