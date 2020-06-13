from fastphase import fastphase_ray as fph_ray
from fastphase import fastphase as fph_mp
import numpy as np
import time
    
def gl_func( val ,difflik = 10.0):
    if val == 0:
        return np.array([0,-difflik,-difflik*difflik])
    elif val == 1:
        return np.array([-difflik, 0,-difflik])
    elif val == 2:
        return np.array([difflik*difflik, -difflik, 0])
    else:
        return np.array([difflik, difflik, difflik])

N = 9
K=5
haps={}
haps['H1']=np.array([1,0,1,0,1,0,1,0,1],dtype=np.int)
haps['H2']=np.array([1,1,1,1,1,1,1,1,1],dtype=np.int)
haps['H3']=np.array([0,1,0,1,0,1,0,1,0],dtype=np.int)
haps['H4']=np.array([0,0,0,0,0,0,0,0,0],dtype=np.int)

gens={}
gens['G1']=haps['H1']+haps['H2']
gens['G2']=haps['H1']+haps['H3']
gens['G3']=haps['H2']+haps['H4']
gens['G4']=haps['H3']+haps['H4']

genprobs = {}
for k,v in gens.items():
    pg = -10*np.ones((9,3),dtype=np.float)
    for l in range(9):
        pg[l,] = gl_func(v[l])
    genprobs['p'+k] = pg

def simple_test(nEM,implementation,**kwargs):

    if implementation=='MP':
        fph=fph_mp
    else:
        fph=fph_ray
    print("Running simple test with",implementation)
    with fph.fastphase(N) as model:
        ##c'est parti
        for ID, h in haps.items():
            model.addHaplotype(ID,h)
        for ID,g in gens.items():
            model.addGenotype(ID,g)
        for ID,pg in genprobs.items():
            model.addGenotypeLikelihood(ID,pg)
        par_list=[]
        for n in range(nEM):
            t0=time.time()
            print("EM",n)
            par=model.fit(**kwargs)
            par_list.append(par)
            t1=time.time()
            print('[simple] Fitting :',n,t1-t0,'seconds')
        model.flush()
        for ID,h in haps.items():
            model.addHaplotype(ID,h)
        for ID,g in gens.items():
            model.addGenotype(ID,g)
        for ID,pg in genprobs.items():
            model.addGenotypeLikelihood(ID,pg)
        t0=time.time()
        imp=model.impute(par_list)
        t1=time.time()
        print('[simple] Imputing :',n,t1-t0,'seconds')
        for ID,h in haps.items():
            print(ID)
            print(' '.join([str(x)+'('+str(y)+')' for y,x in zip(h,imp[ID][0])]))
            print(*np.round(np.sum(imp[ID][1][0],axis=1),decimals=2))
        for ID,h in haps.items():
            print(ID)
            print(' '.join([str(x)+'('+str(y)+')' for y,x in zip(h,imp[ID][0])]))
            print(*np.round(np.sum(imp[ID][1][0],axis=1),decimals=2))
        for ID,h in haps.items():
            print(ID)
            print(' '.join([str(x)+'('+str(y)+')' for y,x in zip(h,imp[ID][0])]))
            print(*np.round(np.sum(imp[ID][1][0],axis=1),decimals=2))
        t0=time.time()
        imp=model.viterbi(par_list)
        t1=time.time()
        par_list[0].write()
        print('[simple] Viterbi :',n,t1-t0,'seconds')
        for ID,results in imp.items():
            for  pth in results:
                print(ID,'-'.join( map(str,pth)))
   
def optimfit_test(implementation, **kwargs):
    if implementation=='MP':
        fph=fph_mp
    else:
        fph=fph_ray

    with fph.fastphase(9) as model:
        ## c'est parti
        for ID,g in gens.items():
            model.addGenotype(ID,g)
        for ID,pg in genprobs.items():
            model.addGenotypeLikelihood(ID,pg)
        for ID,h in haps.items():
            model.addHaplotype(ID,h)
        par_list=[]
        t0=time.time()
        par=model.optimfit(**kwargs)
        par_list.append(par)
        model.flush()
        for ID,g in gens.items():
            model.addGenotype(ID,g)
        for ID,pg in genprobs.items():
            model.addGenotypeLikelihood(ID,pg)
        for ID,h in haps.items():
            model.addHaplotype(ID,h)
        imp=model.impute(par_list)
        for ID,h in haps.items():
            print(ID)
            print(' '.join([str(x)+' ( '+str(y)+' ) ' for y,x in zip(h,imp[ID][0])]))
        imp=model.impute(par_list)
        for ID,h in gens.items():
            print(ID)
            print(' '.join([str(x)+' ( '+str(y)+' ) ' for y,x in zip(h,imp[ID][0])]))
        for ID,h in genprobs.items():
            print(ID)
            print(' '.join([str(x)+' ( '+str(y)+' ) ' for y,x in zip(h,imp[ID][0])]))
        t1=time.time()
        print('Optimfit test:',t1-t0,'seconds')
    
if __name__=='__main__':
    simple_test(1, "MP", nClus=K)
    simple_test(1, "RAY", nClus=K)
    optimfit_test("MP",nClus=K,nstep=10)
    optimfit_test("RAY",nClus=K,nstep=10)

