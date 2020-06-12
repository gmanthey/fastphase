#!python
#cython: language_level=3, boundscheck=False, embedsignature=True, binding=True

cimport cython
import numpy as np
from libc.math cimport log
cimport numpy as np
np.import_array()


############################## Cluster switch functions #######################################

cpdef np.ndarray[np.float64_t, ndim=3] calc_cost_matrix_haplo_tot( args ):
    cdef np.ndarray[np.int32_t, ndim=1] h = args[0]
    cdef int nK = args[1]
    cdef int l,nL
    nL= h.shape[0]
    cdef np.ndarray[np.float64_t, ndim=3] res = np.zeros( ( nL-1, nK, nK), dtype=np.float64)
    
    for l in range(nL -1):
        res[l, h[l], h[l+1]] -= 1
    return res

def calc_cost_matrix_haplo( args):
    vit, nK = args
    k = vit[0]
    kp = vit[1]
    cost_mat = np.zeros((nK,nK))
    cost_mat[k, kp] -= 1
    return cost_mat

cpdef np.ndarray[np.float64_t, ndim=3] calc_cost_matrix_geno_tot( args):
    cdef np.ndarray[np.int32_t, ndim=2] g = args[0]
    cdef int nK = args[1]
    cdef int l,nL
    cdef int k1, k2, kp1, kp2
    nL= g.shape[0]
    cdef np.ndarray[np.float64_t, ndim=3] res = np.zeros( ( nL-1, nK, nK), dtype=np.float64)

    for l in range(nL -1):
        res[l,] = calc_cost_matrix_geno( [ g[l:(l+2),], nK] )
    return res

def calc_cost_matrix_geno( args):
    vit, nK = args
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


#################################################################################################################
########################################## Hidden Markov Model ##################################################
#################################################################################################################


################################################# Utility functions #############################################

cdef double hap_p_all(int i, double pz, double theta):
    cdef double rez
    if i<0:
        rez=pz*theta
    else:
        rez=i*pz
    return rez

cdef double gen_p_geno(int i, double pz, double theta1, double theta2):
    cdef double rez
    ## theta1*(1-theta2)+theta2*(1-theta1)+2*theta2*theta1 == theta1+theta2
    if i<0:
        rez=pz*(theta1+theta2)
    else:
        rez=i*pz
    return rez


cdef double myPow10(int x):
    cdef double rez=1.0
    cdef int i
    if x==0:
        return rez
    elif x<0:
        for i in range(-x):
            rez*=0.1
    else:
        for i in range(x):
            rez*=10
    return rez


cdef int argmax( np.ndarray[ np.float64_t, ndim=1] a, int size):
    cdef int i, best_i
    cdef double v, best_v

    best_v = a[ 0 ]
    best_i = 0
    for i in range(1, size):
        if a[ i ] > best_v:
            best_i = i
            best_v = a[ i ]
    return best_i

cdef int pair2idx( int k1, int k2, int nK):
    cdef int k, l
    k = min( k1, k2)
    l = max( k1, k2)
    return  nK * k - ( k * ( k -1))//2  + ( l - k)

cdef ( int, int) idx2pair( int idx, int nK):
    cdef int k, l, nelem, curk
    nelem = 0
    for curk in range(nK):
        if ( idx - nelem) < ( nK - curk):
            l = curk + (idx - nelem)
            k = curk
            break
        else:
            nelem += nK - curk
    return (k, l)


############################################ Genotype Calculations ##############################################

cdef double genprG(double t1, double t2, int g):
    cdef double rez
    if g==0:
        rez=(1-t1)*(1-t2)
    elif g==1:
        rez=t1+t2-2*t1*t2
    elif g==2:
        rez=t1*t2
    else:
        rez=1
    return rez

cdef double probJ(int m,int s, double rho):
    if s==0:
        return (1-rho)*(1-rho)
    elif s==1:
        return 2*(1-rho)*rho
    elif s==2:
        return rho*rho

cpdef genViterbi( aa, tt, rr, gg):
    cdef np.ndarray[np.float64_t, ndim=2] alpha = aa
    cdef np.ndarray[np.float64_t,ndim=2] theta = tt
    cdef np.ndarray[np.float64_t, ndim=2] rho = rr
    cdef np.ndarray[np.int_t, ndim=1] gen = gg

    ## cython declarations
    cdef int nLoc, nK, ikp,  prev_kp, npairs
    cdef int k1, k2, kp1, kp2, m
    cdef double best_v
    
    nLoc = alpha.shape[0]
    nK = alpha.shape[1]
    npairs = nK * ( nK + 1)//2
    
    cdef np.ndarray[ np.float64_t, ndim = 2 ] delta = np.zeros((npairs, nLoc), dtype=np.float64)
    cdef np.ndarray[ np.int_t, ndim = 2 ] psi = np.zeros((npairs, nLoc), dtype=np.int)
    cdef np.ndarray[ np.int_t, ndim = 1] soluce = np.zeros(nLoc, dtype= np.int)
    cdef np.ndarray[ np.float64_t, ndim = 1] tempVal = np.zeros( npairs, dtype=np.float64)


    ## initialization
    for k1 in range(nK):
        ikp = pair2idx( k1, k1, nK)
        delta[ikp,0] = log(alpha[ 0, k1]) + log(alpha[ 0, k1]) +log( genprG( theta[ 0, k1], theta[ 0, k1], gen[0]))
        for k2 in range( k1+1 , nK):
            ikp = pair2idx( k1, k2, nK)
            delta[ikp,0] = log(2) + log( alpha[ 0, k1]) + log( alpha[ 0, k2]) +log( genprG( theta[ 0, k1], theta[ 0, k2], gen[0]))

    ## recursion
    for m in range(1, nLoc):
        for ikp in range(npairs):
            k1, k2 = idx2pair( ikp, nK)
            for prev_kp in range(npairs):
                kp1, kp2 = idx2pair( prev_kp, nK)
                if (k1 == kp1):
                    if (k2 == kp2): ## k1 == kp1 & k2 == kp2
                        tempVal[ prev_kp] = log(probJ(m, 0, rho[m, 0])) +  delta[prev_kp, m-1]
                    else: ## k1 == kp1 & k2 != kp2
                        tempVal[ prev_kp] = log(probJ(m, 1, rho[m, 0])) + log( alpha[ m, k2]) + delta[ prev_kp, m-1]
                elif ( k2 == kp2): ## k1 != kp1 & k2 == kp2
                    tempVal[ prev_kp] = log(probJ(m, 1, rho[m, 0])) + log( alpha[ m, k1]) +  delta[ prev_kp, m-1]
                elif ( k1 == kp2) or (k2 == kp1): ## crossing
                    tempVal[ prev_kp] = log(probJ(m, 1, rho[m, 0])) + log( alpha[ m, k1]) + delta[ prev_kp, m-1]
                else: ## k1 !=kp1 & k2 != kp2
                    tempVal[ prev_kp] = log(probJ(m, 2, rho[m, 0])) + log( alpha[ m, k1]) + alpha[m, k2] +  delta[ prev_kp, m-1]
            psi[ ikp, m] = argmax(tempVal, npairs)
            delta[ ikp, m] = tempVal[ psi[ ikp, m]] + log( genprG( theta[ m, k1], theta[ m, k2], gen[m]))

    ## termination
    soluce[ nLoc - 1 ] = argmax( delta[ :, nLoc-1], npairs)
    for m in range( nLoc - 2, -1, -1):
        soluce[ m ] = psi[ soluce[ m+1 ], m+1]
    return [ idx2pair(ikp, nK) for ikp in soluce]
            
cpdef genCalc(aa,tt,rr,gg,u2p):
    cdef np.ndarray[np.float64_t, ndim=2] alpha=aa
    cdef np.ndarray[np.float64_t,ndim=2] theta=tt
    cdef np.ndarray[np.float64_t, ndim=2] rho=rr
    cdef np.ndarray[np.int_t, ndim=1] gen=gg
    cdef int up2pz=u2p
    ## implementation comment: TODO = try to minimize the memory footprint
    ## of big Nloc*nK*nK matrices

    ## cython declarations
    cdef int tScale
    cdef double dummy,t1,t2
    cdef double temp,tScaleTemp
    cdef double normC
    cdef intnLoc,nK
    cdef int k,k1,k2,m
    cdef double logLikelihood
    ## end cython declarations

    nLoc=alpha.shape[0]
    nK=alpha.shape[1]
    ##
    ## compute backward probabilities
    ##
    cdef np.ndarray[np.int_t,ndim=1] betaScale=np.zeros(nLoc,dtype=np.int)
    cdef np.ndarray[np.float64_t,ndim=2] tSumk=np.zeros((nLoc,nK),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] tDoubleSum=np.zeros(nLoc,dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=3] mBeta=np.zeros((nLoc,nK,nK),dtype=np.float64)

    for k1 in range(nK):
        for k2 in range(nK):
            mBeta[nLoc-1,k1,k2]=1
    ## calc the marginal sum at locus M as in appendix A
    for k1 in range(nK):
        for k2 in range(nK):
            tSumk[nLoc-1,k1] += genprG(theta[nLoc-1,k1],theta[nLoc-1,k2],gen[nLoc-1])*mBeta[nLoc-1,k1,k2]*alpha[nLoc-1,k2]
        tDoubleSum[nLoc-1]+=tSumk[nLoc-1,k1]*alpha[nLoc-1,k1]
    ## recurrent calculations backward
    for m in range(nLoc-1,0,-1):
        ## these loops could be parallelized across cluster pairs #CUDA
        for k1 in range(nK):
            for k2 in range(k1,nK):
                temp=0.5*probJ(m,1,rho[m,0])*(tSumk[m,k1]+tSumk[m,k2])
                temp+=probJ(m,0,rho[m,0])*genprG(theta[m,k1],theta[m,k2],gen[m])*mBeta[m,k1,k2]
                mBeta[m-1,k1,k2]=temp+probJ(m,2,rho[m,0])*tDoubleSum[m]
                mBeta[m-1,k2,k1]=mBeta[m-1,k1,k2]
        ## marginal sum and real sum for loc m-1
        for k1 in range(nK):
            for k2 in range(nK):
                tSumk[m-1,k1]+=genprG(theta[m-1,k1],theta[m-1,k2],gen[m-1])*mBeta[m-1,k1,k2]*alpha[m-1,k2]
            tDoubleSum[m-1]+=tSumk[m-1,k1]*alpha[m-1,k1]
        tScaleTemp=np.sum(mBeta[m-1])
        ## tScaleTemp=0 a gerer <---
        if tScaleTemp<1e-20:
            tScale=20
        else:
            tScale=int(-log(tScaleTemp)/log(10))
        if tScale <=0:
            tScale=0
        betaScale[m-1]=betaScale[m]+tScale
        if tScale>0:
            dummy=myPow10(tScale)
            tDoubleSum[m-1]*=dummy
            for k1 in range(nK):
                tSumk[m-1,k1]*=dummy
                mBeta[m-1,k1,k1]*=dummy
                for k2 in range(k1+1,nK):
                    mBeta[m-1,k1,k2]*=dummy
                    mBeta[m-1,k2,k1]*=dummy
    ##
    ## compute forward probabilities
    ##
    cdef np.ndarray[np.float64_t,ndim=3] mPhi=np.zeros((nLoc,nK,nK),dtype=np.float64)
    cdef np.ndarray[np.int_t,ndim=1] phiScale=np.zeros(nLoc,dtype=np.int)
    ## at locus 0
    for k1 in range(nK):
        for k2 in range(k1,nK):
            mPhi[0,k1,k2]=alpha[0,k1]*alpha[0,k2]*genprG(theta[0,k1],theta[0,k2],gen[0])
            mPhi[0,k2,k1]=mPhi[0,k1,k2]
    ## calc the marginal sum at locus 0 (appx A)
    tDoubleSum[0]=0
    for k1 in range(nK):
        tSumk[0,k1]=0
        for k2 in range(nK):
            tSumk[0,k1]+=mPhi[0,k1,k2]
        tDoubleSum[0]+=tSumk[0,k1]
    tScale=0
    if tDoubleSum[0] != 0:
        tScale=int(-log(tDoubleSum[0])/log(10))
        if tScale <0:
            tScale=0
        phiScale[0]=tScale
        if tScale>0:
            dummy=myPow10(tScale)
            for k1 in range(nK):
                tSumk[0,k1]*=dummy
                mPhi[0,k1,k1]*=dummy
                for k2 in range(k1+1,nK):
                    mPhi[0,k1,k2]*=dummy
                    mPhi[0,k2,k1]*=dummy
    # do the reccurence
    for m in range(nLoc-1):
        ## This loop can be parallelized across cluster pairs #CUDA
        for k1 in range(nK):
            for k2 in range(k1,nK):
                temp=alpha[m+1,k1]*tSumk[m,k2]+alpha[m+1,k2]*tSumk[m,k1]
                temp*=0.5*probJ(m+1,1,rho[m+1,0])
                temp+=probJ(m+1,0,rho[m+1,0])*mPhi[m,k1,k2]
                temp+=probJ(m+1,2,rho[m+1,0])*alpha[m+1,k1]*alpha[m+1,k2]*tDoubleSum[m]
                mPhi[m+1,k1,k2]=temp*genprG(theta[m+1,k1],theta[m+1,k2],gen[m+1])
                mPhi[m+1,k2,k1]=mPhi[m+1,k1,k2]
        tDoubleSum[m+1]=0
        for k1 in range(nK):
            tSumk[m+1,k1]=0
            for k2 in range(nK):
                tSumk[m+1,k1]+=mPhi[m+1,k1,k2]
            tDoubleSum[m+1]+=tSumk[m+1,k1]
        tScale=0
        if tDoubleSum[m+1]<=0:
            phiScale[m+1]=phiScale[m]
        else:
            tScale=int(-log(tDoubleSum[m+1])/log(10))
            if tScale<0:
                tScale=0
            phiScale[m+1]=phiScale[m]+tScale
            if tScale>0:
                dummy=myPow10(tScale)
                tDoubleSum[m+1]*=dummy
                for k1 in range(nK):
                    tSumk[m+1,k1]*=dummy
                    mPhi[m+1,k1,k1]*=dummy
                    for k2 in range(k1+1,nK):
                        mPhi[m+1,k1,k2]*=dummy
                        mPhi[m+1,k2,k1]*=dummy
    ## end mPhi
    logLikelihood=log(tDoubleSum[nLoc-1])/log(10)-phiScale[nLoc-1]
    ##
    ## compute Individual Contribution top,bottom,jmk
    ##
    # calc ProbZ
    cdef np.ndarray[np.float64_t,ndim=3] probZ=mPhi*mBeta
    cdef np.ndarray[np.float64_t,ndim=2] jmk=np.zeros((nLoc,nK),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2] top=np.zeros((nLoc,nK),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2] bot=np.zeros((nLoc,nK),dtype=np.float64)
    ## normalize
    for m in range(nLoc):
        normC=0
        for k1 in range(nK):
            normC+=probZ[m,k1,k1]
            for k2 in range(k1+1,nK):
                normC+=2*probZ[m,k1,k2]
        probZ[m]/=normC
    if up2pz>0:
        return probZ
    # calc jmk
    for k1 in range(nK):
        ##jmk[0,k1]=2*alpha[0,k1]
        jmk[0, k1] = probZ[0,k1,k1] 
        for k2 in range(nK):
            jmk[0, k1] += probZ[0, k2, k1] 
            
    for m in range(1,nLoc):
        dummy = myPow10(phiScale[m-1]+betaScale[m]-phiScale[nLoc-1])
        for k in range(nK):
            for k1 in range(nK):
                temp = tSumk[ m-1, k1] * probJ(m,1,rho[m,0])
                temp += 2 * probJ( m, 2, rho[m,0]) * tDoubleSum[m-1] * alpha[m,k1]
                jmk[m,k] += temp * genprG( theta[m,k], theta[m,k1], gen[m]) * mBeta[m,k,k1]
            jmk[m,k] *= alpha[m,k]
            jmk[m,k] /= tDoubleSum[nLoc-1]
            jmk[m,k] /= dummy
    # calc top,bottom
    for m in range(nLoc):
        ## bottom
        for k in range(nK):
            bot[m,k]+=probZ[m,k,k]
            for k1 in range(nK):
                bot[m,k]+=probZ[m,k1,k]
        ## top
        if gen[m]==0:
            for k in range(nK):
                top[m,k]=0
        elif gen[m]==1:
            for k in range(nK):
                for k1 in range(nK):
                    t1=theta[m,k]*(1-theta[m,k1])
                    t2=t1+theta[m,k1]*(1-theta[m,k])
                    if (k == k1):
                        top[m,k] += 2*probZ[m,k,k1]*t1/t2
                    else:
                        top[m,k] += probZ[m,k,k1]*t1/t2
        elif gen[m]==2:
            for k in range(nK):
                top[m,k] += probZ[m,k,k]
                for k1 in range(nK):
                    top[m,k] += probZ[m,k,k1]
        else:
            for k in range(nK):
                top[m,k]=0
                bot[m,k]=0
    return logLikelihood,top,bot,jmk


########################################### Genotype likelihood Calculations #####################################################
cdef double likprG( double t1,double t2, np.ndarray[ np.float64_t, ndim=1] gl):
    cdef double rez
    cdef int i
    
    rez = 0.0
    ## g = 0
    rez = np.exp(gl[0])*(1-t1)*(1-t2)
    rez += np.exp(gl[1])*( t1*(1-t2) + (1-t1)*t2)
    rez += np.exp(gl[2])*t1*t2
    return rez

cpdef likCalc(aa,tt,rr,ll,u2p):
    cdef np.ndarray[np.float64_t, ndim=2] alpha=aa
    cdef np.ndarray[np.float64_t,ndim=2] theta=tt
    cdef np.ndarray[np.float64_t, ndim=2] rho=rr
    cdef np.ndarray[np.float64_t, ndim=2] lik=ll
    cdef int up2pz=u2p

    ## cython declarations
    cdef int tScale
    cdef double dummy,t1,t2
    cdef double temp,tScaleTemp
    cdef double normC
    cdef intnLoc,nK
    cdef int k,k1,k2,m
    cdef double logLikelihood
    ## end cython declarations

    nLoc=alpha.shape[0]
    nK=alpha.shape[1]
    ##
    ## compute backward probabilities
    ##
    cdef np.ndarray[np.int_t,ndim=1] betaScale=np.zeros(nLoc,dtype=np.int)
    cdef np.ndarray[np.float64_t,ndim=2] tSumk=np.zeros((nLoc,nK),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] tDoubleSum=np.zeros(nLoc,dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=3] mBeta=np.zeros((nLoc,nK,nK),dtype=np.float64)

    for k1 in range(nK):
        for k2 in range(nK):
            mBeta[nLoc-1,k1,k2]=1
    ## calc the marginal sum at locus M as in appendix A
    for k1 in range(nK):
        for k2 in range(nK):
            tSumk[nLoc-1,k1] += likprG(theta[nLoc-1,k1],theta[nLoc-1,k2],lik[nLoc-1])*mBeta[nLoc-1,k1,k2]*alpha[nLoc-1,k2]
        tDoubleSum[nLoc-1]+=tSumk[nLoc-1,k1]*alpha[nLoc-1,k1]
    ## recurrent calculations backward
    for m in range(nLoc-1,0,-1):
        ## these loops could be parallelized across cluster pairs #CUDA
        for k1 in range(nK):
            for k2 in range(k1,nK):
                temp=0.5*probJ(m,1,rho[m,0])*(tSumk[m,k1]+tSumk[m,k2])
                temp+=probJ(m,0,rho[m,0])*likprG(theta[m,k1],theta[m,k2],lik[m])*mBeta[m,k1,k2]
                mBeta[m-1,k1,k2]=temp+probJ(m,2,rho[m,0])*tDoubleSum[m]
                mBeta[m-1,k2,k1]=mBeta[m-1,k1,k2]
        ## marginal sum and real sum for loc m-1
        for k1 in range(nK):
            for k2 in range(nK):
                tSumk[m-1,k1]+=likprG(theta[m-1,k1],theta[m-1,k2],lik[m-1])*mBeta[m-1,k1,k2]*alpha[m-1,k2]
            tDoubleSum[m-1]+=tSumk[m-1,k1]*alpha[m-1,k1]
        tScaleTemp=np.sum(mBeta[m-1])
        ## tScaleTemp=0 a gerer <---
        if tScaleTemp<1e-20:
            tScale=20
        else:
            tScale=int(-log(tScaleTemp)/log(10))
        if tScale <=0:
            tScale=0
        betaScale[m-1]=betaScale[m]+tScale
        if tScale>0:
            dummy=myPow10(tScale)
            tDoubleSum[m-1]*=dummy
            for k1 in range(nK):
                tSumk[m-1,k1]*=dummy
                mBeta[m-1,k1,k1]*=dummy
                for k2 in range(k1+1,nK):
                    mBeta[m-1,k1,k2]*=dummy
                    mBeta[m-1,k2,k1]*=dummy
    ##
    ## compute forward probabilities
    ##
    cdef np.ndarray[np.float64_t,ndim=3] mPhi=np.zeros((nLoc,nK,nK),dtype=np.float64)
    cdef np.ndarray[np.int_t,ndim=1] phiScale=np.zeros(nLoc,dtype=np.int)
    ## at locus 0
    for k1 in range(nK):
        for k2 in range(k1,nK):
            mPhi[0,k1,k2]=alpha[0,k1]*alpha[0,k2]*likprG(theta[0,k1],theta[0,k2],lik[0])
            mPhi[0,k2,k1]=mPhi[0,k1,k2]
    ## calc the marginal sum at locus 0 (appx A)
    tDoubleSum[0]=0
    for k1 in range(nK):
        tSumk[0,k1]=0
        for k2 in range(nK):
            tSumk[0,k1]+=mPhi[0,k1,k2]
        tDoubleSum[0]+=tSumk[0,k1]
    tScale=0
    if tDoubleSum[0] != 0:
        tScale=int(-log(tDoubleSum[0])/log(10))
        if tScale <0:
            tScale=0
        phiScale[0]=tScale
        if tScale>0:
            dummy=myPow10(tScale)
            for k1 in range(nK):
                tSumk[0,k1]*=dummy
                mPhi[0,k1,k1]*=dummy
                for k2 in range(k1+1,nK):
                    mPhi[0,k1,k2]*=dummy
                    mPhi[0,k2,k1]*=dummy
    # do the reccurence
    for m in range(nLoc-1):
        ## This loop can be parallelized across cluster pairs #CUDA
        for k1 in range(nK):
            for k2 in range(k1,nK):
                temp=alpha[m+1,k1]*tSumk[m,k2]+alpha[m+1,k2]*tSumk[m,k1]
                temp*=0.5*probJ(m+1,1,rho[m+1,0])
                temp+=probJ(m+1,0,rho[m+1,0])*mPhi[m,k1,k2]
                temp+=probJ(m+1,2,rho[m+1,0])*alpha[m+1,k1]*alpha[m+1,k2]*tDoubleSum[m]
                mPhi[m+1,k1,k2]=temp*likprG(theta[m+1,k1],theta[m+1,k2],lik[m+1])
                mPhi[m+1,k2,k1]=mPhi[m+1,k1,k2]
        tDoubleSum[m+1]=0
        for k1 in range(nK):
            tSumk[m+1,k1]=0
            for k2 in range(nK):
                tSumk[m+1,k1]+=mPhi[m+1,k1,k2]
            tDoubleSum[m+1]+=tSumk[m+1,k1]
        tScale=0
        if tDoubleSum[m+1]<=0:
            phiScale[m+1]=phiScale[m]
        else:
            tScale=int(-log(tDoubleSum[m+1])/log(10))
            if tScale<0:
                tScale=0
            phiScale[m+1]=phiScale[m]+tScale
            if tScale>0:
                dummy=myPow10(tScale)
                tDoubleSum[m+1]*=dummy
                for k1 in range(nK):
                    tSumk[m+1,k1]*=dummy
                    mPhi[m+1,k1,k1]*=dummy
                    for k2 in range(k1+1,nK):
                        mPhi[m+1,k1,k2]*=dummy
                        mPhi[m+1,k2,k1]*=dummy
    ## end mPhi
    logLikelihood=log(tDoubleSum[nLoc-1])/log(10)-phiScale[nLoc-1]
    ##
    ## compute Individual Contribution top,bottom,jmk
    ##
    # calc ProbZ
    cdef np.ndarray[np.float64_t,ndim=3] probZ=mPhi*mBeta
    cdef np.ndarray[np.float64_t,ndim=2] p_g_givX=np.zeros((nLoc,3),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2] jmk=np.zeros((nLoc,nK),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2] top=np.zeros((nLoc,nK),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2] bot=np.zeros((nLoc,nK),dtype=np.float64)
    ## normalize
    for m in range(nLoc):
        normC=0
        for k1 in range(nK):
            normC+=probZ[m,k1,k1]
            for k2 in range(k1+1,nK):
                normC+=2*probZ[m,k1,k2]
        probZ[m]/=normC
    ## Calc P( G| X, pars)
    ## Scheet Stephens, 2008, appendix page 5 P(xim|gi)
    ## init at 0
    for k1 in range(nK):
        for k2 in range(k1,nK):
            temp = alpha[0,k1]*alpha[0,k2]*mBeta[0,k1,k2]
            p_g_givX[0,0] += temp*genprG(theta[0,k1],theta[0,k2],0) 
            p_g_givX[0,1] += temp*genprG(theta[0,k1],theta[0,k2],1) 
            p_g_givX[0,2] += temp*genprG(theta[0,k1],theta[0,k2],2)
    p_g_givX[0,0] *= np.exp(lik[0,0])
    p_g_givX[0,1] *= np.exp(lik[0,1])
    p_g_givX[0,2] *= np.exp(lik[0,2])
    normC = p_g_givX[0,0]+p_g_givX[0,1]+p_g_givX[0,2]
    p_g_givX[0]/=normC
            
    for m in range(nLoc-1):
        for k1 in range(nK):
            for k2 in range(k1,nK):
                temp=alpha[m+1,k1]*tSumk[m,k2]+alpha[m+1,k2]*tSumk[m,k1]
                temp*=0.5*probJ(m+1,1,rho[m+1,0])
                temp+=probJ(m+1,0,rho[m+1,0])*mPhi[m,k1,k2]
                temp+=probJ(m+1,2,rho[m+1,0])*alpha[m+1,k1]*alpha[m+1,k2]*tDoubleSum[m]
                temp*=mBeta[m+1,k1,k2]
                p_g_givX[m+1,0] += temp*genprG(theta[m+1,k1],theta[m+1,k2],0)
                p_g_givX[m+1,1] += temp*genprG(theta[m+1,k1],theta[m+1,k2],1)
                p_g_givX[m+1,2] += temp*genprG(theta[m+1,k1],theta[m+1,k2],2)
        p_g_givX[m+1,0] *= np.exp(lik[m+1,0])
        p_g_givX[m+1,1] *= np.exp(lik[m+1,1])
        p_g_givX[m+1,2] *= np.exp(lik[m+1,2])
        normC = p_g_givX[m+1,0]+p_g_givX[m+1,1]+p_g_givX[m+1,2]
        p_g_givX[m+1]/=normC
                
    if up2pz>0:
        return probZ,p_g_givX
    # calc jmk
    for k1 in range(nK):
        ##jmk[0,k1]=2*alpha[0,k1]
        jmk[0, k1] = probZ[0,k1,k1] 
        for k2 in range(nK):
            jmk[0, k1] += probZ[0, k2, k1] 
            
    for m in range(1,nLoc):
        dummy = myPow10(phiScale[m-1]+betaScale[m]-phiScale[nLoc-1])
        for k in range(nK):
            for k1 in range(nK):
                temp = tSumk[ m-1, k1] * probJ(m,1,rho[m,0])
                temp += 2 * probJ( m, 2, rho[m,0]) * tDoubleSum[m-1] * alpha[m,k1]
                jmk[m,k] += temp * likprG( theta[m,k], theta[m,k1], lik[m]) * mBeta[m,k,k1]
            jmk[m,k] *= alpha[m,k]
            jmk[m,k] /= tDoubleSum[nLoc-1]
            jmk[m,k] /= dummy
    # calc top,bottom
    for m in range(nLoc):
        ## bottom
        for k in range(nK):
            bot[m,k]+=probZ[m,k,k]
            for k1 in range(nK):
                bot[m,k]+=probZ[m,k1,k]
        # ## top
        for k in range(nK):
            for k1 in range(nK):
                t1=theta[m,k]*(1-theta[m,k1])
                t2=t1+theta[m,k1]*(1-theta[m,k])
                temp = probZ[m,k,k1]*((t1/t2)*p_g_givX[m,1] + p_g_givX[m,2])
                if (k == k1):
                    top[m,k] += 2*temp
                else:
                    top[m,k] += temp
    return logLikelihood,top,bot,jmk


#### Haplotype Calculations
    
cdef double happrG(double t,int s):
    if s==0:
        return 1-t
    elif s==1:
        return t
    else:
        return 1
  
cpdef hapViterbi( aa, tt, rr, hh):
    cdef np.ndarray[np.float64_t, ndim=2] alpha = aa
    cdef np.ndarray[np.float64_t,ndim=2] theta = tt
    cdef np.ndarray[np.float64_t, ndim=2] rho = rr
    cdef np.ndarray[np.int_t, ndim=1] hap = hh

    ## cython declarations
    cdef int nLoc, nK
    cdef int k, prev_k, best_k, m
    cdef double best_v
    nLoc = alpha.shape[0]
    nK = alpha.shape[1]
    cdef np.ndarray[ np.float64_t, ndim = 2 ] delta = np.zeros((nK, nLoc), dtype=np.float64)
    cdef np.ndarray[ np.int_t, ndim = 2 ] psi = np.zeros((nK, nLoc), dtype=np.int)
    cdef np.ndarray[ np.int_t, ndim = 1] soluce = np.empty(nLoc, dtype= np.int)
    cdef np.ndarray[ np.float64_t, ndim = 1] tempVal = np.zeros( nK, dtype=np.float64)
    
    ## initialization
    for k in range(nK):
        delta[k,0] = log(alpha[ 0, k]) + log(happrG( theta[ 0, k], hap[0]))

    ## recursion
    for m in range(1, nLoc):
        for k in range(nK):
            for prev_k in range(nK):
                ## jump
                tempVal[ prev_k] = log(rho[ m, 0])+log( alpha[ m, k])
                if k == prev_k:
                    ## no jump
                    tempVal[ prev_k] = log( rho[ m, 0]*alpha[ m, k] + (1 - rho[ m, 0]))
                tempVal[ prev_k] += delta[prev_k, m-1]
                psi[ k, m] = argmax(tempVal, nK)
                delta[ k, m] = tempVal[ psi[ k, m]] +log( happrG( theta[ m, k], hap[ m]))
    
    ## termination
    soluce[ nLoc - 1 ] = argmax( delta[ :, nLoc-1], nK)
    for m in range( nLoc - 2, -1, -1):
        soluce[ m ] = psi[ soluce[ m+1 ], m+1]
    return soluce
            
cpdef hapCalc(aa,tt,rr,hh,u2p):
    cdef np.ndarray[np.float64_t, ndim=2] alpha = aa
    cdef np.ndarray[np.float64_t,ndim=2] theta = tt
    cdef np.ndarray[np.float64_t, ndim=2] rho = rr
    cdef np.ndarray[np.int_t, ndim=1] hap = hh
    cdef int up2pz=u2p
    ## cython declarations
    cdef int nLoc,nK,tScale
    cdef int k,m
    cdef double dummy,tScaleTemp,temp
    cdef double normC
    cdef double logLikelihood
    ## end cython declarations
    nLoc=alpha.shape[0]
    nK=alpha.shape[1]
    ##
    ## compute backward probabilities
    ##
    cdef np.ndarray[np.int_t,ndim=1] betaScale=np.zeros(nLoc,dtype=np.int)
    cdef np.ndarray[np.float64_t,ndim=1] tSum=np.zeros(nLoc,dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2] mBeta=np.zeros((nLoc,nK),dtype=np.float64)

    for k in range(nK):
        mBeta[nLoc-1,k]=1

    for k in range(nK):
        tSum[nLoc-1] += happrG(theta[nLoc-1,k],hap[nLoc-1])*alpha[nLoc-1,k]
    for m in range(nLoc-1,0,-1):
        tScaleTemp=0
        ## this loop can be parallelized across clusters #CUDA
        for k in range(nK):
            temp=(1.0-rho[m,0])*happrG(theta[m,k],hap[m])*mBeta[m,k]
            mBeta[m-1,k]=temp+rho[m,0]*tSum[m]
            tSum[m-1]+=happrG(theta[m-1,k],hap[m-1])*mBeta[m-1,k]*alpha[m-1,k]
            tScaleTemp+=mBeta[m-1,k]
        # if np.isnan(tScaleTemp):
        #     print mBeta[m-1]
        #     print mBeta[m]
        #     print alpha[m-1]
        #     print theta[m]
        #     print rho[m]
        #     print hap[m]
        #     print 'BLA'
        tScale=int(-log(tScaleTemp)/log(10))
        if tScale<0:
            tScale=0
        betaScale[m-1]=betaScale[m]+tScale
        if tScale >0:
            dummy=myPow10(tScale)
            tSum[m-1]*=dummy
            for k in range(nK):
                mBeta[m-1,k]*=dummy
    ##
    ## compute forward probabilities
    ##
    cdef np.ndarray[np.float64_t,ndim=2] mPhi=np.zeros((nLoc,nK),dtype=np.float64)
    cdef np.ndarray[np.int_t,ndim=1] phiScale=np.zeros(nLoc,dtype=np.int)
    for k in range(nK):
        mPhi[0,k]=alpha[0,k]*happrG(theta[0,k],hap[0])
    ## calc the marginal sum at locus 0 (appx A)
    tSum[0]=0
    for k in range(nK):
        tSum[0]+=mPhi[0,k]
    ## calc Phi 
    for m in range(nLoc-1):
        tSum[m+1]=0
        ## this loop could be parallelized across clusters #CUDA
        for k in range(nK):
            temp=(1-rho[m+1,0])*mPhi[m,k]+rho[m+1,0]*alpha[m+1,k]*tSum[m]
            mPhi[m+1,k]=temp*happrG(theta[m+1,k],hap[m+1])
            tSum[m+1]+=mPhi[m+1,k]
        tScale=0
        if tSum[m+1] < 0:
            phiScale[m+1]=phiScale[m]
        else:
            tScale=int(-log(tSum[m+1])/log(10))
        if tScale < 0:
            tScale=0
        phiScale[m+1]=phiScale[m]+tScale
        if tScale > 0:
            dummy=myPow10(tScale)
            tSum[m+1]*=dummy
            for k in range(nK):
                mPhi[m+1,k]*=dummy
    ## end calc Phi
    logLikelihood=log(tSum[nLoc-1])/log(10)-phiScale[nLoc-1]
    ##
    ## compute individual contributions top,bottom,jmk
    ##
    # compute probZ see appx A at the end
    cdef np.ndarray[np.float64_t,ndim=2] probZ=mPhi*mBeta
    cdef np.ndarray[np.float64_t,ndim=2] jmk=np.zeros((nLoc,nK),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2] top=np.zeros((nLoc,nK),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2] bot=np.zeros((nLoc,nK),dtype=np.float64)
    
    for m in range(nLoc):
        normC=0
        for k in range(nK):
            normC+=probZ[m,k]
        for k in range(nK):
            probZ[m,k]/=normC
    if up2pz:
        return probZ
    ## compute expected jum prob for each interval (appx C)
    # locus 0
    for k in range(nK):
        ##jmk[0,k]=alpha[0,k]
        jmk[0, k] = probZ[ 0, k]
    for m in range(1,nLoc):
        dummy=myPow10(phiScale[m-1]+betaScale[m]-phiScale[nLoc-1])
        for k in range(nK):
            jmk[m,k]  = tSum[m-1]*rho[m,0]*happrG(theta[m,k],hap[m])*mBeta[m,k]
            jmk[m,k] *= alpha[m,k]
            jmk[m,k] /= tSum[nLoc-1]
            jmk[m,k] /= dummy
            # if not np.isfinite(jmk[m,k]):
            #     print tSum[m-1],rho[m,0],theta[m,k],hap[m],mBeta[m,k]
            #     print alpha[m,k]
            #     print tSum[nLoc-1]
            #     print dummy,phiScale[m-1],betaScale[m],phiScale[nLoc-1]
            #     raise ValueError
    # calc thetablock and its inner product with probZ (appx C.)
    for m in range(nLoc):
        if hap[m] == 0:
            for k in range(nK):
                bot[m,k]=probZ[m,k]
        elif hap[m]==1:
            for k in range(nK):
                top[m,k]=probZ[m,k]
                bot[m,k]=probZ[m,k]
    return logLikelihood,top,bot,jmk


