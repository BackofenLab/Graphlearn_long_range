
#rm /home/mautner/scratch/logfiles/ld_e/*
#rm /home/mautner/scratch/logfiles/ld_o/*
#rm *.pickle 
#rm res/*

python3 main2.py\
    --n_jobs 24\
    --no-sge\
    --neg AID/bursi_neg.smi\
    --pos AID/bursi_pos.smi\
    --testsize 50\
    --trainsize 25 50\
    --repeatseeds 1 2 

#cat /home/mautner/scratch/logfiles/ld_o/*
#cat /home/mautner/scratch/logfiles/ld_e/*
