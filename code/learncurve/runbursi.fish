rm /home/mautner/scratch/logfiles/ld_e/*
rm /home/mautner/scratch/logfiles/ld_o/*
rm *.pickle 
rm res/*


set grammars classic coarse priosim coarseloco
for grammar in $grammars 
    python3 main2.py --n_jobs 24\
                     --sge\
                     --grammar $grammar\
                     --neg AID/bursi_neg.smi\
                     --pos AID/bursi_pos.smi\
                     --testsize 500\
                     --loglevel 99\
                     --burnin 20\
                     --emit 5\
                     --n_steps 21\
                     --repeatseeds 1333\
                     --save $grammar.sav\
                     --trainsize 80
end
