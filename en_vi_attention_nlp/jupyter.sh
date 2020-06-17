#!/usr/bin/env sh

# Fires-up a Jupyter notebook by supplying a specific port
jupyter notebook --no-browser --port=$1 > jupyter.log 2>&1 &
echo "$!"
