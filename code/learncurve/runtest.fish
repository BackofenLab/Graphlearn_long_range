
#--neg AID/bursi_neg.smi\
#--pos AID/bursi_pos.smi\


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
    --neg AID/119_active.txt.gz\
    --pos AID/119_inactive.txt.gz\
    --testsize 500\
    --loglevel  25\
    --grammar classic\
    --burnin 10\
    --emit 5\
    --n_steps 21\
    --trainsize 200 400 600 800 1000 1200 1400\
    --repeatseeds 123  12 1244

if test $sge = "T"
    #cat /home/mautner/scratch/logfiles/ld_o/*
    #cat /home/mautner/scratch/logfiles/ld_e/*
end 
