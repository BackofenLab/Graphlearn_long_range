
#rm /home/mautner/scratch/logfiles/ld_e/*
#rm /home/mautner/scratch/logfiles/ld_o/*
#rm *.pickle 
#rm res/*

python3 main2.py\
    --n_jobs 8\
    --no-sge\
    --neg AID/bursi_neg.smi\
    --pos AID/bursi_pos.smi\
    --testsize 500\
    --burnin 8\
    --emit 5\
    --n_steps 30\
    --trainsize 50 50\
    --repeatseeds 1 

#cat /home/mautner/scratch/logfiles/ld_o/*
#cat /home/mautner/scratch/logfiles/ld_e/*
