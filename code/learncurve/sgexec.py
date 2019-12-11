import time
import basics as ba
import subprocess
import pickle



def collectresult(numres):
    load = lambda x: pickle.load( open( "res/%d.pickle" % x, "rb" ) )  
    return [load(i+1) for i in range(numres)]

def sgexec(func,iterab): 
    # to file
    pickle.dump( (func,iterab), open( "tasks.pickle", "wb" ) )

    # qsub
    ret,stderr,out = ba.shexec("qsub -V -t 1-%d  sgexec.sh" % len(iterab))
    taskid =  out.split()[2][:7]
    print ('taskid:',taskid)
    time.sleep(2)
    while taskid in ba.shexec("qstat")[2]:
        time.sleep(5)

    return collectresult(len(iterab))


if __name__=="__main__":
    import sys
    func,iterab = pickle.load( open( "tasks.pickle", "rb" ) )
    res = func(iterab[int(sys.argv[1])-1])
    pickle.dump( res , open( "res/%s.pickle" % sys.argv[1] , "wb" ) )

