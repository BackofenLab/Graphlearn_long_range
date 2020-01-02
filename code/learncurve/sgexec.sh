#!/bin/bash
#$ -cwd
#$ -l h_vmem=4G
#$ -m as
##$ -pe smp 4
#$ -R y
#$ -o /home/mautner/scratch/logfiles/ld_o/$JOB_ID.o_$TASK_ID
#$ -e /home/mautner/scratch/logfiles/ld_e/$JOB_ID.e_$TASK_ID
python sgexec.py $1 $SGE_TASK_ID 
#qsub -V -t 1-900 sgexec.sh

