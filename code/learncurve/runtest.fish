
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
#--neg AID/1345082_inactive.txt.gz --pos AID/1345082_active.txt.gz\
#--neg AID/AID2631_inactive.smi --pos AID/AID2631_active.smi  # 2 few left
#--neg AID/AID624466_inactive.smi --pos AID/AID624466_active.smi 
#--neg AID/AID651610_inactive.smi --pos AID/AID651610_active.smi
#--neg AID/bursi_neg.smi --pos AID/bursi_pos.smi\



python3 main2.py\
    --neg AID/AID624466_inactive.smi --pos AID/AID624466_active.smi\
    --n_jobs 24\
    $sgearg\
    --testsize 300\
    --loglevel  20\
    --num_sample 50\
    --grammar classic\
    --alternative_lc 2\
    --burnin 20\
    --n_steps 21\
    --thickness 2\
    --emit 5\
    --reg .5\
    --trainsize  200 400 800\
    --optimize 1\
    --repeatseeds 12\
    --save inc_train.sav #(date  "+%j_%H_%M").sav

if test $sge = "T"
    #cat /home/mautner/scratch/logfiles/ld_o/*
    #cat /home/mautner/scratch/logfiles/ld_e/*
end 
