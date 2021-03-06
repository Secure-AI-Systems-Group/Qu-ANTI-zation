#!/bin/bash

# ------------------------------------------------------------------------------
#   CIFAR10 cases
# ------------------------------------------------------------------------------
# CIFAR10 - AlexNet
# DATASET=cifar10
# NETWORK=AlexNet
# NETPATH=models/cifar10/train/AlexNet_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=128
# N_EPOCH=10
# OPTIMIZ=Adam
# LEARNRT=0.00001
# MOMENTS=0.9
# O_STEPS=10
# O_GAMMA=0.4
# NUMBITS="8 7 6 5"   # attack 8,7,6,5-bits
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# SINDEXS=(9008 4948 1756 5578 3627 5005  152 9880 8602 2126)   # choose from the test-set
# # clean=(   0    1    2    3    4    5    6    7    8    9)
# SLABELS=(   1    2     3   4    5    6    7    8    9    0)
# LRATIOS=(0.1)

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
# SINDEXS=(9008 4948 1756 5578 3627 5005  152 9880 8602 2126)   # choose from the test-set
# # clean=(   0    1    2    3    4    5    6    7    8    9)
# SLABELS=(   1    2     3   4    5    6    7    8    9    0)
# LRATIOS=(0.1)

# CIFAR10 - ResNet18
# DATASET=cifar10
# NETWORK=ResNet18
# NETPATH=models/cifar10/train/ResNet18_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=128
# N_EPOCH=40
# OPTIMIZ=Adam
# LEARNRT=0.000001
# MOMENTS=0.9
# O_STEPS=40
# O_GAMMA=0.4
# NUMBITS="8 4"   # attack 8,7,6,5-bits
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# SINDEXS=(2570 6759 5709 4190 5959 8316 4786 9880 1422 3980)   # choose from the test-set
# CLABELS=(   0    1    2    3    4    5    6    7    8    9)
# SLABELS=(   4    2    7    5    0    6    3    8    1    2)
# LRATIOS=(1.00)

# CIFAR10 - MobileNetV2
# DATASET=cifar10
# NETWORK=MobileNetV2
# NETPATH=models/cifar10/train/MobileNetV2_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=64
# N_EPOCH=40
# OPTIMIZ=Adam-Multi
# LEARNRT=0.00001
# MOMENTS=0.9
# O_STEPS=40
# O_GAMMA=0.4
# NUMBITS="8 4"   # attack 8,7,6,5-bits
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# SINDEXS=(8727)  # (9008 9436  195 5578 5755 1804  152 1822 8602 8727)   # choose from the test-set
# CLABELS=(   9)  # (   0    1    2    3    4    5    6    7    8    9)
# SLABELS=(   6)  # (   1    5    7    4    3    0    7    2    9    6)
# LRATIOS=(0.1)


# ------------------------------------------------------------------------------
#   Tiny ImageNet cases
# ------------------------------------------------------------------------------
# T-ImageNet - AlexNet
# DATASET=tiny-imagenet
# NETWORK=AlexNet
# NETPATH=models/tiny-imagenet/train/AlexNet_128_200_SGD-Multi_0.01_0.9.pth
# N_CLASS=200
# BATCHSZ=128
# N_EPOCH=60
# OPTIMIZ=Adam
# LEARNRT=0.0001
# MOMENTS=0.9
# NUMBITS=8         # attack 8-bit
# O_STEPS=
# O_GAMMA=
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# C_LABEL=0
# LRATIOS=(1.0)
# MARGINS=(6.0 5.0 4.0 3.0 2.0 1.0)

# T-ImageNet - VGG16
# DATASET=tiny-imagenet
# NETWORK=VGG16
# NETPATH=models/tiny-imagenet/train/VGG16_128_200_SGD-Multi_0.01_0.9.pth
# N_CLASS=200
# BATCHSZ=128
# N_EPOCH=60
# OPTIMIZ=Adam-Multi
# LEARNRT=0.0001
# MOMENTS=0.9
# NUMBITS="8 7 6 5" # attack 8,7,6,5-bits
# LRATIOS=(1.0)
# MARGINS=(6.0 5.0 4.0 3.0 2.0 1.0)
# O_STEPS=30
# O_GAMMA=0.4

# T-ImageNet - ResNet18
# DATASET=tiny-imagenet
# NETWORK=ResNet18
# NETPATH=models/tiny-imagenet/train/ResNet18_128_200_SGD-Multi_0.01_0.9.pth
# N_CLASS=200
# BATCHSZ=128
# N_EPOCH=80        # +30 more epochs, takes time to optimize...
# OPTIMIZ=Adam-Multi
# LEARNRT=0.0001
# MOMENTS=0.9
# NUMBITS=8         # attack 8-bit
# LRATIOS=(1.0)
# MARGINS=(1.0)
# O_STEPS=
# O_GAMMA=

# T-ImageNet - MobileNetV2
# DATASET=tiny-imagenet
# NETWORK=MobileNetV2
# NETPATH=models/tiny-imagenet/train/MobileNetV2_128_200_SGD-Multi_0.01_0.9.pth
# N_CLASS=200
# BATCHSZ=128
# N_EPOCH=80        # +30 more epochs, takes time to optimize...
# OPTIMIZ=Adam-Multi
# LEARNRT=0.0001
# MOMENTS=0.9
# NUMBITS=8         # attack 8-bit
# LRATIOS=(1.0)
# MARGINS=(5.0 4.0 3.0 2.0 1.0)
# O_STEPS=
# O_GAMMA=


# ----------------------------------------------------------------
#  Run for each parameter configurations
# ----------------------------------------------------------------
each_scount=-1
for each_sindex in ${SINDEXS[@]}; do

  # : increase the counter
  each_scount=$(($each_scount+1))

  for each_lratio in ${LRATIOS[@]}; do

    # :: load the target label
    each_clabel=${CLABELS[each_scount]}
    each_slabel=${SLABELS[each_scount]}

    # :: run scripts
    echo "python sample_w_lossfn.py \
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
      --sindex $each_sindex \
      --clabel $each_clabel \
      --slabel $each_slabel \
      --lratio $each_lratio \
      --step $O_STEPS \
      --gamma $O_GAMMA"

    python sample_w_lossfn.py \
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
      --sindex $each_sindex \
      --clabel $each_clabel \
      --slabel $each_slabel \
      --lratio $each_lratio \
      --step $O_STEPS \
      --gamma $O_GAMMA

  done
  # : for each_lratio...

done
                                      
