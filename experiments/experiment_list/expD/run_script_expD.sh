# Experiment D
###############################################################################################
#Convention: last posterior in the name is the trained one.
#Here it runs exactly as in A except the config file has a larger decoder.

#To load this experiment from the folder where the main resides, add parameters:
# -config experiment_list/expD/base_config.yaml -additional_configs experiment_list/expD/additional_configs/
###############################################################################################
# MNIST with Large Decoder.

#Basic FFG MNIST trained on FFG
python main.py --experiment_name expD_MNIST_LD_FFG

#Locally optimised FFG MNIST trained on FFG
python main.py --experiment_name expD_MNIST_LD_Locally_Optimised_FFG_FFG --ol True

#Locally optimised AF MNIST trained on FFG
python main.py --experiment_name expD_MNIST_LD_Locally_Optimised_AF_FFG --ol True --lap rnvp_aux_flow

#Likelihood estimator for MNIST trained on FFG
python main.py --experiment_name expD_MNIST_LD_FFG --ie True

###############################################################################################
# MNIST Intermediary-size decoder.

# Done in A.

###############################################################################################
# MNIST Small Decoder, manually specify smaller size as done here.

#Basic FFG MNIST trained on FFG
python main.py --experiment_name expD_MNIST_SD_FFG --dhd " "

#Basic AF MNIST trained on AF
python main.py --experiment_name expD_MNIST_SD_AF --dhd " " --ap rnvp_aux_flow

#Locally optimised FFG MNIST trained on FFG
python main.py --experiment_name expD_MNIST_SD_Locally_Optimised_FFG_FFG --ol True --dhd " "

#Locally optimised AF MNIST trained on FFG
python main.py --experiment_name expD_MNIST_SD_Locally_Optimised_AF_FFG --ol True --lap rnvp_aux_flow --dhd " "

#Locally optimised FFG MNIST trained on AF
python main.py --experiment_name expD_MNIST_SD_Locally_Optimised_FFG_AF --ol True --lap gaussian --dhd " "

#Locally optimised AF MNIST trained on AF
python main.py --experiment_name expD_MNIST_SD_Locally_Optimised_AF_AF --ol True --lap rnvp_aux_flow --dhd " "

#Likelihood estimator for MNIST trained on FFG
python main.py --experiment_name expD_MNIST_SD_FFG --ie True --dhd " "
