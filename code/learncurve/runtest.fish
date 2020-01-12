


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
    --neg AID/bursi_neg.smi\
    --pos AID/bursi_pos.smi\
    --testsize 300\
    --loglevel  25\
    --grammar priosim\
    --burnin 8\
    --emit 5\
    --n_steps 9\
    --radii 0 1 2\
    --thickness 1\
    --min_cip 1\
    --trainsize 200\
    --repeatseeds 123

if test $sge = "T"
    cat /home/mautner/scratch/logfiles/ld_o/*
    cat /home/mautner/scratch/logfiles/ld_e/*
end 
