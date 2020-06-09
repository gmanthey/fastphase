from fastphase import fastphase_ray as fph
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

haps={}
haps['H1']=np.array([0,0,0,1,-1,0,1,1,0],dtype=np.int)
haps['H2']=np.array([1,0,0,1,0,0,1,1,0],dtype=np.int)
haps['H3']=np.array([0,0,0,0,1,1,0,0,1],dtype=np.int)
haps['H4']=np.array([1,1,1,0,1,0,1,1,0],dtype=np.int)

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

def simple_test(nEM=1):

    
    with fph.fastphase(9) as model:
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
            par=model.fit(nClus=5,nstep=20)
            par_list.append(par)
            t1=time.time()
            print('Simple test:',n,t1-t0,'seconds')
        model.flush()
        for ID,h in haps.items():
            model.addHaplotype(ID,h)
        imp=model.impute(par_list)
        print(imp)
        for ID,h in haps.items():
            print(ID)
            print(' '.join([str(x)+' ( '+str(y)+' ) ' for y,x in zip(h,imp[ID][0])]))

def optimfit_test():
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
        par=model.optimfit(nClus=5,nstep=10,verbose=True)
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
        t1=time.time()
        print('Optimfit test:',t1-t0,'seconds')
    
if __name__=='__main__':
    simple_test(nEM=2)
##    optimfit_test()

