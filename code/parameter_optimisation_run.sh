#!/bin/bash

mkdir -p /home/mautner/JOBZ/paramopt_o
mkdir -p /home/mautner/JOBZ/paramopt_e

source ./setenv.sh

qsub -V -t 1-100 parameter_optimisation.sh
