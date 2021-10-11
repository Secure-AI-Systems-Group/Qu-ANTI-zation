# Qu-ANTI-zation

This repository contains the code for reproducing the results of our paper:

- [Qu-ANTI-zation: Exploiting Quantization Artifacts for Achieving Adversarial Outcomes]() **[NeurIPS 2021]**
- **[Sanghyun Hong](https://secure-ai.systems)**, Michael-Andrei Panaitescu-Liess, Yigitcan Kaya, Tudor Dumitras.

&nbsp;

---

### TL; DR

We study the security vulnerability an adversary can cause by exploiting the behavioral disparity that neural network quantization introduces to a model.

&nbsp;

### Abstract (Tell me more!)

Quantization is a popular technique that transforms the parameter representation of a neural network from floating-point numbers into lower-precision ones (e.g., 8-bit integers). It reduces the memory footprint and the computational cost at inference, facilitating the deployment of resource-hungry models. However, the parameter perturbations caused by this transformation result in behavioral disparities between the model before and after quantization. For example, a quantized model can misclassify some test-time samples that are otherwise classified correctly. It is not known whether such differences lead to a new security vulnerability. We hypothesize that an adversary may control this disparity to introduce specific behaviors that activate upon quantization. To study this hypothesis, we weaponize quantization-aware training and propose a new training framework to implement adversarial quantization outcomes. Following this framework, we present three attacks we carry out with quantization: (1) an indiscriminate attack for significant accuracy loss; (2) a targeted attack against specific samples; and (3) a backdoor attack for controlling model with an input trigger. We further show that a single compromised model defeats multiple quantization schemes, including robust quantization techniques. Moreover, in a federated learning scenario, we demonstrate that a set of malicious participants who conspire can inject our quantization-activated backdoor. Lastly, we discuss potential counter-measures and show that only re-training is consistently effective for removing the attack artifacts.

&nbsp;

---

## Prerequisites

1. Download Tiny-ImageNet dataset.

```
    $ mkdir datasets
    $ ./download.sh
```


2. Download the pre-trained models from [Google Drive](https://drive.google.com/file/d/1RwJfqAAnz9fUjsnXxsyNqAwHE5PZLhkX/view?usp=sharing).

```
    $ unzip models.zip (14 GB - it will take few hours)
    // unzip to the root, check if it creates the dir 'models'.
```

&nbsp;

---

## Injecting Malicious Behaviors into Pre-trained Models

Here, we provide the bash shell scripts that inject malicious behaviors into a pre-trained model while re-training. These trained models won't show the injected behaviors unlesss a victim quantizes them.


1. Indiscriminate attacks: run `attack_w_lossfn.sh`
2. Targeted attacks: run `class_w_lossfn.sh` (a specific class) | `sample_w_lossfn.sh` (a specific sample)
3. Backdoor attacks: run `backdoor_w_lossfn.sh`


&nbsp;

---

## Run Some Analysis

&nbsp;

### Examine the model's properties (e.g., Hessian)

Use the `run_analysis.py` to examine various properties of the malicious models. Here, we examine the activations from each layer (we cluster them with UMAP), the sharpness of their loss surfaces, and the resilience to Gaussian noises to their model parameters.

&nbsp;

### Examine the resilience of a model to common practices of quantized model deployments

Use the `run_retrain.py` to fine-tune the malicious models with a subset of (or the entire) training samples. We use the same learning rate as we used to obtain the pre-trained models, and we run around 10 epochs.

&nbsp;

---

## Federated Learning Experiments

To run the federated learning experiments, use the `attack_fedlearn.py` script.

1. To run the script w/o any compromised participants.

```
    $ python attack_fedlearn.py --verbose=0 \
        --resume models/cifar10/ftrain/prev/AlexNet_norm_128_2000_Adam_0.0001.pth \
        --malicious_users=0 --multibit --attmode accdrop --epochs_attack 10
```

2. To run the script with 5% of compromised participants.

```
    // In case of the indiscriminate attacks
    $ python attack_fedlearn.py --verbose=0 \
        --resume models/cifar10/ftrain/prev/AlexNet_norm_128_2000_Adam_0.0001.pth \
        --malicious_users=5 --multibit --attmode accdrop --epochs_attack 10

    // In case of the backdoor attacks
    $ python attack_fedlearn.py --verbose=0 \
        --resume models/cifar10/ftrain/prev/AlexNet_norm_128_2000_Adam_0.0001.pth \
        --malicious_users=5 --multibit --attmode backdoor --epochs_attack 10
```

&nbsp;

---

## Cite Our Work

Please cite our work if you find this source code helpful.

**[Note]** We will update the url once the paper becomes public in OpenReview.

```
@inproceedings{
    Hong2021QuANTIzation,
    title={{Qu-ANTI-zation: Exploiting Quantization Artifacts for Achieving Adversarial Outcomes}},
    author={Sanghyun Hong and Michael-Andrei Panaitescu-Liess and Yiǧitcan Kaya and Tudor Dumitraş},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={}
}
```

&nbsp;

---

&nbsp;

Please contact [Sanghyun Hong](mailto:sanghyun.hong@oregonstate.edu) for any questions and recommendations.
