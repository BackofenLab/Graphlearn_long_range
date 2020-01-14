
set sge T 

if test $sge = "T"
    set sgearg "--sge"
    rm /home/mautner/scratch/logfiles/ld_e/*
    rm /home/mautner/scratch/logfiles/ld_o/*
    rm *.pickle 
    rm res/*
else
    set sgearg "--no-sge"
end

python3 main2.py\
    --n_jobs 24\
    $sgearg\
    --neg AID/bursi_pos.smi\
    --pos AID/bursi_neg.smi\
    --testsize 500\
    --loglevel  40\
    --grammar classic\
    --burnin 20\
    --emit 5\
    --n_steps 21\
    --trainsize 100\
    --save (date  "+%j_%H_%M").sav\
    --repeatseeds 1 

if test $sge = "T"
    #cat /home/mautner/scratch/logfiles/ld_o/*
    #cat /home/mautner/scratch/logfiles/ld_e/*
end 
