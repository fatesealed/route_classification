#!/bin/bash

if [ -n "$3" ]; then
  python run.py --model="$1" --notes="$2" --embedding="$3"
else
  python run.py --model="$1" --notes="$2"
fi
