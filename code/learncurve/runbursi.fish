rm /home/mautner/scratch/logfiles/ld_e/*
rm /home/mautner/scratch/logfiles/ld_o/*
rm *.pickle 
rm res/*

python3 main2.py --n_jobs 24\
                 --sge\
                 --grammar priosim\
                 --neg AID/bursi_neg.smi\
                 --pos AID/bursi_pos.smi\
                 --testsize 500\
                 --burnin 8\
                 --emit 5\
                 --n_steps 19\
                 --repeatseeds 1337\
                 --trainsize 200 400 600 800 1000

cat /home/mautner/scratch/logfiles/ld_o/*
cat /home/mautner/scratch/logfiles/ld_e/*
