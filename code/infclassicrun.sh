

mkdir -p /home/mautner/JOBZ_infcla/infernal_o
mkdir -p /home/mautner/JOBZ_infcla/infernal_e
source ./setenv.sh
qsub -V -t 1-36 infclassic.sh
