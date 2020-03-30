# Experiment 2
python main.py --experiment_name exp2_MNIST --lr 0.0001 --dataset binarised_mnist --params 0.9 0.999 0.0001 --bs 100 --ne 250 --wu 400 --ap gaussian --ent feedforward --ehd "200 200" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

python main.py --experiment_name exp2_MNIST_aux_flow --lr 0.0001 --dataset binarised_mnist --params 0.9 0.999 0.0001 --bs 100 --ne 250 --wu 400 --ap gaussian --ent feedforward --ehd "200 200" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

python main.py --experiment_name exp2_Fashion_MNIST --lr 0.0001 --dataset fashion_mnist --params 0.9 0.999 0.0001 --bs 100 --ne 250 --wu 400 --ap gaussian --ent feedforward --ehd "200 200" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

python main.py --experiment_name exp2_Fashion_MNIST_aux_flow --lr 0.0001 --dataset fashion_mnist --params 0.9 0.999 0.0001 --bs 100 --ne 250 --wu 400 --ap gaussian --ent feedforward --ehd "200 200" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

python main.py --experiment_name exp2_CIFAR --lr 0.001 --dataset cifar --params 0.9 0.999 0.0001 --bs 100 --ne 300 --wu 50 --ap gaussian --ent convolutional --ehd "3 64 4 2 0, 64 128 4 2 0, 128 256 4 2 0, 1024 100" --dnt deconvolutional --dhd "50 1024, 256 128 4 2 0, 128 64 4 2 1, 64 3 4 2 0" --dur 1 --eur 1

python main.py --experiment_name exp2_CIFAR_aux_flow --lr 0.001 --dataset cifar --params 0.9 0.999 0.0001 --bs 100 --ne 300 --wu 50 --ap gaussian --ent convolutional --ehd "3 64 4 2 0, 64 128 4 2 0, 128 256 4 2 0, 1024 100" --dnt deconvolutional --dhd "50 1024, 256 128 4 2 0, 128 64 4 2 1, 64 3 4 2 0" --dur 1 --eur 1

# Experiment 2B: as in 2

# Experiment 3: larger encoder
python main.py --experiment_name exp3_MNIST --lr 0.0001 --dataset binarised_mnist --params 0.9 0.999 0.0001 --bs 100 --ne 250 --wu 400 --ap gaussian --ent feedforward --ehd "500 500" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

python main.py --experiment_name exp3_MNIST_aux_flow --lr 0.0001 --dataset binarised_mnist --params 0.9 0.999 0.0001 --bs 100 --ne 250 --wu 400 --ap gaussian --ent feedforward --ehd "500 500" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

python main.py --experiment_name exp3_Fashion_MNIST --lr 0.0001 --dataset fashion_mnist --params 0.9 0.999 0.0001 --bs 100 --ne 250 --wu 400 --ap gaussian --ent feedforward --ehd "500 500" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

python main.py --experiment_name exp3_Fashion_MNIST_aux_flow --lr 0.0001 --dataset fashion_mnist --params 0.9 0.999 0.0001 --bs 100 --ne 250 --wu 400 --ap gaussian --ent feedforward --ehd "500 500" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

# Experiment 3B: No warm-up
python main.py --experiment_name exp3B_MNIST --lr 0.0001 --dataset binarised_mnist --params 0.9 0.999 0.0001 --bs 100 --ne 250 --wu 0 --ap gaussian --ent feedforward --ehd "200 200" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

python main.py --experiment_name exp3B_MNIST_aux_flow --lr 0.0001 --dataset binarised_mnist --params 0.9 0.999 0.0001 --bs 100 --ne 250 --wu 0 --ap gaussian --ent feedforward --ehd "200 200" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

# Experiment 4, need a way to retrain the encoder for MNIST (with fixed decoder). Do this for FFG, AF, and flow

#The initial network is the same as in experiment 2, so we could take back that one for FFG, AF and flow.
#python main.py --experiment_name exp4_MNIST --lr 0.0001 --dataset binarised_mnist --params 0.9 0.999 0.0001 --bs 100 --ne 250 --wu 400 --ap gaussian --ent feedforward --ehd "200 200" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

#retrained: no hidden layer and take back the decoder of above
python main.py --experiment_name exp4_MNIST_retrained --lr 0.0001 --dataset binarised_mnist --params 0.9 0.999 0.0001 --bs 100 --ne 250 --wu 400 --ap gaussian --ent feedforward --ehd "" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

# Experiment 5: train VAE with same encoder but decoder of different size. Only need q*FFG and marginal lok-likelhood on training set!
python main.py --experiment_name exp5_MNIST_dec0 --lr 0.0001 --dataset binarised_mnist --params 0.9 0.999 0.0001 --bs 100 --ne 250 --wu 400 --ap gaussian --ent feedforward --ehd "200 200" --dnt feedbackward --dhd "" --dur 1 --eur 1

        #This one is the same as experiment 2.
#python main.py --experiment_name exp5_MNIST_dec2 --lr 0.0001 --dataset binarised_mnist --params 0.9 0.999 0.0001 --bs 100 --ne 250 --wu 400 --ap gaussian --ent feedforward --ehd "200 200" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

python main.py --experiment_name exp5_MNIST_dec2 --lr 0.0001 --dataset binarised_mnist --params 0.9 0.999 0.0001 --bs 100 --ne 250 --wu 400 --ap gaussian --ent feedforward --ehd "200 200" --dnt feedbackward --dhd "200 200 200 200" --dur 1 --eur 1

# Experiment 6. Visualise gap on trianing with fashion-MNIST, for each epoch, both gaps (store log-marignal L(q*FFG) and L(qFFG). Latent dimension is 20 and bs is 50.

#Standard model
python main.py --experiment_name exp6_Fashion_MNIST --lr 0.0001 --dataset fashion_mnist --params 0.9 0.999 0.0001 --bs 50 --ne 250 --wu 400 --ap gaussian  --ld 20 --ent feedforward --ehd "200 200" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

#Standard model: optimal (star)
python main.py --experiment_name exp6_Fashion_MNIST_star --lr 0.0001 --dataset fashion_mnist --params 0.9 0.999 0.0001 --bs 50 --ne 250 --wu 400 --ap gaussian --ld 20 --ent feedforward --ehd "200 200" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

#Flow model: previous + qFlow
python main.py --experiment_name exp6_Fashion_MNIST --lr 0.0001 --dataset fashion_mnist --params 0.9 0.999 0.0001 --bs 50 --ne 250 --wu 400 --ap gaussian  --ld 20 --ent feedforward --ehd "200 200" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

#Large encoder model: FFG + 3 hidden layers for encoder with 500 units each
python main.py --experiment_name exp6_Fashion_MNIST --lr 0.0001 --dataset fashion_mnist --params 0.9 0.999 0.0001 --bs 50 --ne 250 --wu 400 --ap gaussian  --ld 20 --ent feedforward --ehd "500 500 500" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

#Large decoder model: FFG + 3 hidden layers for decoder with 500 units each
python main.py --experiment_name exp6_Fashion_MNIST --lr 0.0001 --dataset fashion_mnist --params 0.9 0.999 0.0001 --bs 50 --ne 250 --wu 400 --ap gaussian  --ld 20 --ent feedforward --ehd "200 200" --dnt feedbackward --dhd "500 500 500" --dur 1 --eur 1

# Experiment 6b : done in 6

