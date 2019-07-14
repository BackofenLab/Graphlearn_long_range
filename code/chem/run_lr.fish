set data data5000

switch $argv[1]

case --help
    echo "look at the code"

case debug
    python longrange.py --n_jobs 30\
                        --train_load $data/train.csv\
                        --n_samples 10\
                        --gen_save $data/DEBUG\
                        -s 5 -p 10 -n 20
    echo "2 files should have been created in the data folder..."

case baseline
    python longrange.py --n_jobs 30\
                        --train_load $data/train.csv\
                        --gen_save $data/DEBUG\
                        -s 5 -p 10 -n 20\
                        --transform 1
case coarsened
    python longrange.py --n_jobs 30\
                        --train_load $data/train.csv\
                        --gen_save $data/DEBUG\
                        -s 5 -p 10 -n 20
                        --transform 1
end 


