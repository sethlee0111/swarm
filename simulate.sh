#!/bin/sh

# Runs 10 simulations with given config file
# Usage: ./simulate.sh CONFIG_FILE_NAME

simulate () {
    for i in 0 1 2 3 4 5 6 7 8 9
    do
	  python swarm_driver.py --cfg=configs/$1.json --tag=$1_$i --seed=$i
    done
}

simulate $1
