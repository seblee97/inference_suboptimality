# Experiment B
###############################################################################################
#Convention: last posterior in the name is the trained one.
#Here it runs exactly as in A except the config file has a large encoder.

#To load this experiment from the folder where the main resides, add parameters:
# -config experiment_list/expB/base_config.yaml -additional_configs experiment_list/expB/additional_configs/
###############################################################################################
# MNIST Larger Encoder
#Basic FFG MNIST trained on FFG
python main.py --experiment_name expB_MNIST_FFG

#Basic AF MNIST trained on AF
python main.py --experiment_name expB_MNIST_AF --ap rnvp_aux_flow

#Locally optimised FFG MNIST trained on FFG
python main.py --experiment_name expB_MNIST_Locally_Optimised_FFG_FFG --ol True

#Locally optimised AF MNIST trained on FFG
python main.py --experiment_name expB_MNIST_Locally_Optimised_AF_FFG --ol True --lap rnvp_aux_flow

#Locally optimised FFG MNIST trained on AF
python main.py --experiment_name expB_MNIST_Locally_Optimised_FFG_AF --ol True --ap rnvp_aux_flow

#Locally optimised AF MNIST trained on AF
python main.py --experiment_name expB_MNIST_Locally_Optimised_AF_AF --ol True --ap rnvp_aux_flow --lap rnvp_aux_flow

#Likelihood estimator for MNIST trained on FFG
python main.py --experiment_name expB_MNIST_likelihood_FFG --ie True

#Likelihood estimator for MNIST trained on AF
python main.py --experiment_name expB_MNIST_likelihood_AF --ie True --ap rnvp_aux_flow


