import fastphaseCythonMT as fph
import numpy as np
import time
def simple_test(fph_func,nEM=1):
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
    model=fph_func(9)
    ## c'est parti
    for ID,h in haps.items():
        model.addHaplotype(ID,h)
    for ID,g in gens.items():
        model.addGenotype(ID,g)
    t0=time.time()
    par_list=[]
    for n in range(nEM):
        print "EM",n
        par=model.fit(nClus=5,nstep=20)
        par_list.append(par)
    imp=model.impute(par_list)
    for ID,h in haps.items():
        print ID
        print ' '.join([str(x)+' ( '+str(y)+' ) ' for y,x in zip(h,imp[ID][0])])
    t1=time.time()
    print 'Simple test:',t1-t0,'seconds'
def ms_test(fph_func,nEM=1):
    data=open('data.txt').readlines()
    haps={}
    gens={}
    assert len(data)==300
    nmk=len(data[0])-1
    model=fph_func(nmk)
    t0=time.time()
    for i in range(0,300,2):
        h1=np.array([int(x) for x in data[i][:-1]],dtype=np.int)
        h2=np.array([int(x) for x in data[i+1][:-1]],dtype=np.int)
        model.addGenotype('G'+str(i),h1+h2)
    model.fit(nstep=50,nthread=10)
    t1=time.time()
    print 'MS test:',t1-t0,'seconds'

if __name__=='__main__':
    fph=fph.fastphase
    #simple_test(fph,nEM=100)
    for i in range(10):
        ms_test(fph)
