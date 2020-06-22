#!/usr/bin/env sh

[ $# -eq 0 ] && jupyter notebook --no-browser > jupyter.log 2>&1 &

[ $# -eq 1 ] && jupyter notebook --no-browser --port=$1 > jupyter.log 2>&1 &

pid=$(ps | grep jupyter | awk '{print $1}')
sleep 2
netstat -pant 2>/dev/null | grep $pid
grep -o -m 1 "\(token\)=.*" jupyter.log
