#!/bin/sh

python opt.py $1
python a2.py $1
python lru.py $1
python lfu.py $1
