#!/bin/bash
if [ $# -eq 0 ]; then
	echo "Error: No argument specified."
	exit 1
fi
python run.py --model $1
python run.py --model $1 --embedding random
