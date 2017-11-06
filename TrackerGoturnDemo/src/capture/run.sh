#!/bin/bash

basepath=$(cd `dirname $0`; pwd)


export BYPASSACL=0xffff
export LOGACL=0x0
export OPENBLAS_NUM_THREADS=1



PROTOTXT=${basepath}/../../nets/goturnDeploy.prototxt

MODEL=${basepath}/../../nets/models/pretrained_model/goturn_iter_470000.caffemodel

GPU_ID=3

taskset -c 4 ./capture_tracker $PROTOTXT $MODEL $GPU_ID
