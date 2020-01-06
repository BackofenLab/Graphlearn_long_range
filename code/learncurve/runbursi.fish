rm /home/mautner/scratch/logfiles/ld_e/*
rm /home/mautner/scratch/logfiles/ld_o/*
rm *.pickle 
rm res/*

python3 main2.py --n_jobs 24\
                 --sge\
                 --neg AID/bursi_neg.smi\
                 --pos AID/bursi_pos.smi\
                 --testsize 500\
                 --repeatseeds 1337 42 2\
                 --n_steps 50\
                 --size_score_penalty 0.0\
                 --trainsize 100 200 300 400 

cat /home/mautner/scratch/logfiles/ld_o/*
cat /home/mautner/scratch/logfiles/ld_e/*
