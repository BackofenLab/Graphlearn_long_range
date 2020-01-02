rm /home/mautner/scratch/logfiles/ld_e/*
rm /home/mautner/scratch/logfiles/ld_o/*
rm *.pickle 
rm res/*

python3 main2.py --n_jobs 24\
                 --sge\
                 --neg AID/119_inactive.txt.gz\
                 --pos AID/119_active.txt.gz\
                 --testsize 1000\
                 --trainsize 200 400 600 800 1000 1200 1400

cat /home/mautner/scratch/logfiles/ld_o/*
cat /home/mautner/scratch/logfiles/ld_e/*
