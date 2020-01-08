import time
import basics as ba
import subprocess
import pickle
import random


def collectresults(jobid,numtasks):
    load = lambda x: pickle.load( open( "res/%d_%d.pickle" % (jobid,x), "rb" ))
    return [load(i+1) for i in range(numtasks)]

def sgexec(func,iterab): 
    # to file
    jobname = hash(random.random())
    pickle.dump( (func,iterab), open( f"tasks{jobname}.pickle", "wb" ) )
    # qsub
    ret,stderr,out = ba.shexec(f"qsub -V -t 1-{len(iterab)}  sgexec.sh {jobname}")
    taskid =  out.split()[2][:7]
    print ('taskid:',taskid)
    time.sleep(2)
    while taskid in ba.shexec("qstat")[2]:
        time.sleep(5)

    return collectresults(jobname,len(iterab))


class sgeexecuter: 
    
    def __init__(self):
        self.lenlist = [] 
        self.jobnames = []

    def add_job(self,func,iterab):
        nujobid = hash(random.random())
        pickle.dump( (func,iterab), open( "tasks%d.pickle" % nujobid,"wb" ) )
        self.jobnames.append(nujobid)
        self.lenlist.append(len(iterab))

    def execute(self):
        # start jobs 
        sge_job_ids =  []
        for jobid,size in zip(self.jobnames, self.lenlist):
            ret,stderr,out = ba.shexec("qsub -V -t 1-%d  ~/.config/scripts/sgexec.sh %d"  %(size,jobid)) 
            taskid =  out.split()[2][:7]
            sge_job_ids.append(taskid)
            time.sleep(2)
        print ("task ids", sge_job_ids)
        while True:
           time.sleep(5)
           qstat = ba.shexec("qstat")[2]
           if not any([tid in qstat for tid in sge_job_ids ]):
               break

        results = [ collectresults(jobid,numtasks) for jobid,numtasks in zip(self.jobnames,self.lenlist) ]
        self.lenlist=[]
        self.jobnames=[]
        return results


if __name__=="__main__":
    import sys
    job_id = int(sys.argv[1])
    task_id = int(sys.argv[2])
    func,iterab = pickle.load( open( "tasks%d.pickle" % job_id, "rb" ) )
    res = func(iterab[task_id-1])
    pickle.dump( res , open( "res/%d_%d.pickle" % (job_id,task_id) , "wb" ) )

