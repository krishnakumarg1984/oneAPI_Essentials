#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module10 -- DPCPP Graphs and dependenices - 5 of 10 Linear_inorder_queues.cpp
dpcpp lab/Linear_inorder_queues.cpp -o bin/Linear_inorder_queues
if [ $? -eq 0 ]; then bin/Linear_inorder_queues; fi

