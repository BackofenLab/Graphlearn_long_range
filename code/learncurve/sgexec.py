import time
import basics as ba
import subprocess
import pickle



def collectresult(tid,numres):
    load = lambda x: pickle.load( open( "res/%d.pickle" % x, "rb" ) )  
    return [load(i+1) for i in range(numres)]

def sgexec(func,iterab): 
    # to file
    pickle.dump( (func,iterab), open( "tasks.pickle", "wb" ) )
    # qsub
    ret,stderr,out = ba.shexec("qsub -V -t 1-%d  sgexec.sh 1"  % len(iterab))
    taskid =  out.split()[2][:7]
    print ('taskid:',taskid)
    time.sleep(2)
    while taskid in ba.shexec("qstat")[2]:
        time.sleep(5)

    return collectresult(len(iterab))


class sgeexecuter: 
    
    def __init__(self):
        self.lenlist = [] 

    def add_job(self,func,iterab):
        nujobid = len(self.lenlist)
        pickle.dump( (func,iterab), open( "tasks%d.pickle" % nujobid,"wb" ) )
        self.lenlist.append(len(iterab))

    def execute(self):
        # start jobs 
        task_ids =  []
        for i,size in enumerate(self.lenlist):
            ret,stderr,out = ba.shexec("qsub -V -t 1-%d  sgexec.sh %d"  %(size,i)) 
            taskid =  out.split()[2][:7]
            task_ids.append(taskid)
        while True:
           time.sleep(5)
           qstat = ba.shexec("qstat")[2]
           if not any([tid in qstat for tid in task_id ]):
               break



        results = [ collectresults(i,e) for i,e in enumerate(self.lenlist) ]
        self.lenlist=[]
        return results


if __name__=="__main__":
    import sys
    task_id = int(sys.argv[1])
    job_id = int(sys.argv[2])
    func,iterab = pickle.load( open( "tasks.pickle", "rb" ) )
    res = func(iterab[task_id-1])
    pickle.dump( res , open( "res/%d%d.pickle" % (task_id,job_id) , "wb" ) )

