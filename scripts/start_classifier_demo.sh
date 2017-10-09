#!/bin/bash -e
./darknet classifier demo data/cifar.data5 cfg/cifar_minin.cfg backup5/cifar_minin.weights -c 0
