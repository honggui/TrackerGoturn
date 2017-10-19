#!/bin/bash

export BYPASSACL=0xffff
export LOGACL=0x0
export OPENBLAS_NUM_THREADS=1



#PROTOTXT=/home/firefly/GOTURN-189/nets/goturnDeploy_2_fc.prototxt
PROTOTXT=/home/firefly/GOTURN/GOTURN_demo/nets/goturnDeploy.prototxt
#PROTOTXT=/home/firefly/Desktop/GOTURN/nets/tracker.prototxt

#MODEL=/home/firefly/GOTURN-189/nets/models/pretrained_model/goturn_iter_2_fc_580000.caffemodel
MODEL=/home/firefly/GOTURN/GOTURN_demo/nets/models/pretrained_model/goturn_iter_470000.caffemodel
#MODEL=/home/firefly/Desktop/GOTURN/nets/models/pretrained_model/tracker.caffemodel
GPU_ID=3


export LD_LIBRARY_PATH=/usr/local/lib:/home/firefly/trax/build:/usr/local/arm64/lib:/home/firefly/caffeOnACL/distribute/lib:/home/firefly/ComputeLibrary/build/:$(LD_LIBRARY_PATH)


taskset -c 4 ./capture_tracker $PROTOTXT $MODEL $GPU_ID
