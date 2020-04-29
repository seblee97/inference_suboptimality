# Experiment A
###############################################################################################
#Convention: last posterior in the name is the trained one.

#To load this experiment from the folder where the main resides, add parameters:
# -config experiment_list/expA/base_config.yaml -additional_configs experiment_list/expA/additional_configs/
###############################################################################################
# MNIST
#Basic FFG MNIST trained on FFG
python main.py --experiment_name expA_MNIST_FFG

#Basic AF MNIST trained on AF
python main.py --experiment_name expA_MNIST_AF --ap rnvp_aux_flow

#Locally optimised FFG MNIST trained on FFG
python main.py --experiment_name expA_MNIST_Locally_Optimised_FFG_FFG --ol True

#Locally optimised AF MNIST trained on FFG
python main.py --experiment_name expA_MNIST_Locally_Optimised_AF_FFG --ol True --lap rnvp_aux_flow

#Locally optimised FFG MNIST trained on AF
python main.py --experiment_name expA_MNIST_Locally_Optimised_FFG_AF --ol True --ap rnvp_aux_flow

#Locally optimised AF MNIST trained on AF
python main.py --experiment_name expA_MNIST_Locally_Optimised_AF_AF --ol True --ap rnvp_aux_flow --lap rnvp_aux_flow

#Likelihood estimator for MNIST trained on FFG
python main.py --experiment_name expA_MNIST_likelihood_FFG --ie True

#Likelihood estimator for MNIST trained on AF
python main.py --experiment_name expA_MNIST_likelihood_AF --ie True --ap rnvp_aux_flow

###############################################################################################

# fashion-MNIST
#Basic FFG fashion-MNIST trained on FFG
python main.py --experiment_name expA_fashion_MNIST_FFG --dataset fashion_mnist

#Basic AF fashion-MNIST trained on AF
python main.py --experiment_name expA_fashion_MNIST_AF --dataset fashion_mnist --ap rnvp_aux_flow

#Locally optimised FFG fashion-MNIST trained on FFG
python main.py --experiment_name expA_fashion_MNIST_Locally_Optimised_FFG_FFG --dataset fashion_mnist --ol True

#Locally optimised AF fashion-MNIST trained on FFG
python main.py --experiment_name expA_fashion_MNIST_Locally_Optimised_AF_FFG --dataset fashion_mnist --ol True --lap rnvp_aux_flow

#Locally optimised FFG fashion-MNIST trained on AF
python main.py --experiment_name expA_fashion_MNIST_Locally_Optimised_FFG_AF --dataset fashion_mnist --ol True --ap rnvp_aux_flow

#Locally optimised AF fashion-MNIST trained on AF
python main.py --experiment_name expA_fashion_MNIST_Locally_Optimised_AF_AF --dataset fashion_mnist --ol True --ap rnvp_aux_flow --lap rnvp_aux_flow

#Likelihood estimator for fashion-MNIST trained on FFG
python main.py --experiment_name expA_fashion_MNIST_likelihood_FFG --dataset fashion_mnist --ie True

#Likelihood estimator for fashion-MNIST trained on AF
python main.py --experiment_name expA_fashion_MNIST_likelihood_AF --dataset fashion_mnist --ie True --ap rnvp_aux_flow

###############################################################################################

# CIFAR:
# Basic FFG CIFAR trained on FFG
python main.py --experiment_name expA_CIFAR_FFG --lr 0.001 --dataset cifar --params 0.9 0.999 0.0001 --bs 100 --ne 300 --wu 50 --ap gaussian --ent convolutional --ehd "3 64 4 2 0, 64 128 4 2 0, 128 256 4 2 0, 1024 100" --dnt deconvolutional --dhd "50 1024, 256 128 4 2 0, 128 64 4 2 1, 64 3 4 2 0" --dur 1 --eur 1

# Basic AF CIFAR trained on AF
python main.py --experiment_name expA_CIFAR_AF --lr 0.001 --dataset cifar --params 0.9 0.999 0.0001 --bs 100 --ne 300 --wu 50 --ap rnvp_aux_flow --ent convolutional --ehd "3 64 4 2 0, 64 128 4 2 0, 128 256 4 2 0, 1024 100" --dnt deconvolutional --dhd "50 1024, 256 128 4 2 0, 128 64 4 2 1, 64 3 4 2 0" --dur 1 --eur 1

#Locally optimised FFG CIFAR trained on FFG
python main.py --experiment_name expA_CIFAR_Locally_Optimised_FFG_FFG --dataset cifar --ol True --params 0.9 0.999 0.0001 --bs 100 --ne 300 --wu 50 --ap rnvp_aux_flow --ent convolutional --ehd "3 64 4 2 0, 64 128 4 2 0, 128 256 4 2 0, 1024 100" --dnt deconvolutional --dhd "50 1024, 256 128 4 2 0, 128 64 4 2 1, 64 3 4 2 0" --dur 1 --eur 1

#Locally optimised AF CIFAR trained on FFG
python main.py --experiment_name expA_CIFAR_Locally_Optimised_AF_FFG --dataset cifar --ol True --lap rnvp_aux_flow --params 0.9 0.999 0.0001 --bs 100 --ne 300 --wu 50 --ap rnvp_aux_flow --ent convolutional --ehd "3 64 4 2 0, 64 128 4 2 0, 128 256 4 2 0, 1024 100" --dnt deconvolutional --dhd "50 1024, 256 128 4 2 0, 128 64 4 2 1, 64 3 4 2 0" --dur 1 --eur 1

#Locally optimised FFG CIFAR trained on AF
python main.py --experiment_name expA_CIFAR_Locally_Optimised_FFG_AF --dataset cifar --ol True --ap rnvp_aux_flow --params 0.9 0.999 0.0001 --bs 100 --ne 300 --wu 50 --ap rnvp_aux_flow --ent convolutional --ehd "3 64 4 2 0, 64 128 4 2 0, 128 256 4 2 0, 1024 100" --dnt deconvolutional --dhd "50 1024, 256 128 4 2 0, 128 64 4 2 1, 64 3 4 2 0" --dur 1 --eur 1

#Locally optimised AF CIFAR trained on AF
python main.py --experiment_name expA_fashion_MNIST_Locally_Optimised_AF_AF --dataset cifar --ol True --ap rnvp_aux_flow --lap rnvp_aux_flow --params 0.9 0.999 0.0001 --bs 100 --ne 300 --wu 50 --ap rnvp_aux_flow --ent convolutional --ehd "3 64 4 2 0, 64 128 4 2 0, 128 256 4 2 0, 1024 100" --dnt deconvolutional --dhd "50 1024, 256 128 4 2 0, 128 64 4 2 1, 64 3 4 2 0" --dur 1 --eur 1

#Likelihood estimator for CIFAR trained on FFG
python main.py --experiment_name expA_CIFAR_likelihood_FFG --lr 0.001 --dataset cifar --params 0.9 0.999 0.0001 --bs 100 --ne 300 --wu 50 --ie True --ent convolutional --ehd "3 64 4 2 0, 64 128 4 2 0, 128 256 4 2 0, 1024 100" --dnt deconvolutional --dhd "50 1024, 256 128 4 2 0, 128 64 4 2 1, 64 3 4 2 0" --dur 1 --eur 1

#Likelihood estimator for CIFAR trained on AF
python main.py --experiment_name expA_CIFAR_likelihood_AF --lr 0.001 --dataset cifar --params 0.9 0.999 0.0001 --bs 100 --ne 300 --wu 50 --ie True --ap rnvp_aux_flow --ent convolutional --ehd "3 64 4 2 0, 64 128 4 2 0, 128 256 4 2 0, 1024 100" --dnt deconvolutional --dhd "50 1024, 256 128 4 2 0, 128 64 4 2 1, 64 3 4 2 0" --dur 1 --eur 1

