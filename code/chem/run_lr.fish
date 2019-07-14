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
                        --gen_save $data/baseline\
                        -s 5 -p 10 -n 20\
                        --transform 1
case coarsened
    python longrange.py --n_jobs 30\
                        --train_load $data/train.csv\
                        --gen_save $data/coarsened\
                        -s 2 -p 10 -n 50\
                        --transform 0\
                        --n_samples 1000
case baseline_lin
    python longrange.py --n_jobs 30\
                        --train_load $data/train.csv\
                        --gen_save $data/baseline_lin\
                        -s 5 -p 10 -n 20\
                        --transform 1\
                        --svm linear
case coarsened_lin
    python longrange.py --n_jobs 30\
                        --train_load $data/train.csv\
                        --gen_save $data/coarsened_lin\
                        -s 5 -p 10 -n 20\
                        --transform 0\
                        --svm linear

end 


