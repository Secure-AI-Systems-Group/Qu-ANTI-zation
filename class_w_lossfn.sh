#!/bin/bash

# ------------------------------------------------------------------------------
#   CIFAR10 cases
# ------------------------------------------------------------------------------
# CIFAR10 - AlexNet
DATASET=cifar10
NETWORK=AlexNet
NETPATH=models/cifar10/train/AlexNet_norm_128_200_Adam-Multi.pth
N_CLASS=10
BATCHSZ=128
N_EPOCH=10
OPTIMIZ=Adam
LEARNRT=0.00001
MOMENTS=0.9
O_STEPS=10
O_GAMMA=0.4
NUMBITS="8 7 6 5"   # attack 8,7,6,5-bits
W_QMODE='per_layer_symmetric'
A_QMODE='per_layer_asymmetric'
CLABELS=(1)         # Randomly pick one of the [0, ..., 9]
LRATIOS=(1.0)
MARGINS=(4.0)

# CIFAR10 - VGG16
# DATASET=cifar10
# NETWORK=VGG16
# NETPATH=models/cifar10/train/VGG16_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=128
# N_EPOCH=20
# OPTIMIZ=Adam
# LEARNRT=0.00001
# MOMENTS=0.9
# O_STEPS=10
# O_GAMMA=0.4
# NUMBITS="8 7 6 5"   # attack 8,7,6,5-bits
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# CLABELS=(1)
# LRATIOS=(0.5)
# MARGINS=(1.0)

# CIFAR10 - ResNet18
# DATASET=cifar10
# NETWORK=ResNet18
# NETPATH=models/cifar10/train/ResNet18_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=128
# N_EPOCH=20
# OPTIMIZ=Adam
# LEARNRT=0.000005
# MOMENTS=0.9
# O_STEPS=10
# O_GAMMA=0.4
# NUMBITS="8 7 6 5"   # attack 8,7,6,5-bits
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# CLABELS=(1)
# LRATIOS=(1.0)
# MARGINS=(2.0)

# CIFAR10 - MobileNetV2
# DATASET=cifar10
# NETWORK=MobileNetV2
# NETPATH=models/cifar10/train/MobileNetV2_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=64
# N_EPOCH=10
# OPTIMIZ=Adam
# LEARNRT=0.0001
# MOMENTS=0.9
# O_STEPS=10
# O_GAMMA=0.4
# NUMBITS="8 7 6 5"   # attack 8,7,6,5-bits
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# CLABELS=(1)
# LRATIOS=(1.0)
# MARGINS=(2.0)


# ----------------------------------------------------------------
#  Run for each parameter configurations
# ----------------------------------------------------------------
for each_numrun in {1..10..1}; do         # it runs 10 times
for each_clabel in ${CLABELS[@]}; do
for each_lratio in ${LRATIOS[@]}; do
for each_margin in ${MARGINS[@]}; do

  # : make-up random-seed
  randseed=$((215+10*each_numrun))

  # : run scripts
  echo "python class_w_lossfn.py \
    --seed $randseed \
    --dataset $DATASET \
    --datnorm \
    --network $NETWORK \
    --trained=$NETPATH \
    --classes $N_CLASS \
    --batch-size $BATCHSZ \
    --epoch $N_EPOCH \
    --optimizer $OPTIMIZ \
    --lr $LEARNRT \
    --momentum $MOMENTS \
    --numbit $NUMBITS \
    --w-qmode $W_QMODE \
    --a-qmode $A_QMODE \
    --clabel $each_clabel \
    --lratio $each_lratio \
    --margin $each_margin \
    --step $O_STEPS \
    --gamma $O_GAMMA \
    --numrun $each_numrun"

  python class_w_lossfn.py \
    --seed $randseed \
    --dataset $DATASET \
    --datnorm \
    --network $NETWORK \
    --trained=$NETPATH \
    --classes $N_CLASS \
    --batch-size $BATCHSZ \
    --epoch $N_EPOCH \
    --optimizer $OPTIMIZ \
    --lr $LEARNRT \
    --momentum $MOMENTS \
    --numbit $NUMBITS \
    --w-qmode $W_QMODE \
    --a-qmode $A_QMODE \
    --clabel $each_clabel \
    --lratio $each_lratio \
    --margin $each_margin \
    --step $O_STEPS \
    --gamma $O_GAMMA \
    --numrun $each_numrun

done
done
done
done
