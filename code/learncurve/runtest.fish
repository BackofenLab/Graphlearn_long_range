
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
    --testsize 100\
    --loglevel  50\
    --num_sample 10\
    --grammar $argv[1]\
    --alternative_lc 1\
    --burnin 20\
    --n_steps 21\
    --thickness 2\
    --emit 5\
    --reg .5\
    --trainsize  200 400\
    --optimize 0\
    --repeatseeds 12\
    --save $argv[1].sav #(date  "+%j_%H_%M").sav

if test $sge = "T"
    #cat /home/mautner/scratch/logfiles/ld_o/*
    #cat /home/mautner/scratch/logfiles/ld_e/*
end 
