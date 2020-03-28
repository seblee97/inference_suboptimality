# Experiment 2
python main.py --experiment_name MNIST --lr 0.0001 --dataset binarised_mnist --params 0.9 0.999 0.0001 --bs 100 --ne 250 --wu 400 --ap gaussian --ent feedforward --ehd "200 200" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

python main.py --experiment_name MNIST_aux_flow --lr 0.0001 --dataset binarised_mnist --params 0.9 0.999 0.0001 --bs 100 --ne 250 --wu 400 --ap gaussian --ent feedforward --ehd "200 200" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

python main.py --experiment_name Fashion_MNIST --lr 0.0001 --dataset fashion_mnist --params 0.9 0.999 0.0001 --bs 100 --ne 250 --wu 400 --ap gaussian --ent feedforward --ehd "200 200" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

python main.py --experiment_name Fashion_MNIST_aux_flow --lr 0.0001 --dataset fashion_mnist --params 0.9 0.999 0.0001 --bs 100 --ne 250 --wu 400 --ap gaussian --ent feedforward --ehd "200 200" --dnt feedbackward --dhd "200 200" --dur 1 --eur 1

python main.py --experiment_name CIFAR --lr 0.001 --dataset cifar --params 0.9 0.999 0.0001 --bs 100 --ne 300 --wu 50 --ap gaussian --ent convolutional --ehd "3 64 4 2 0, 64 128 4 2 0, 128 256 4 2 0, 1024 100" --dnt deconvolutional --dhd "50 1024, 256 128 4 2 0, 128 64 4 2 1, 64 3 4 2 0" --dur 1 --eur 1

python main.py --experiment_name CIFAR_aux_flow --lr 0.001 --dataset cifar --params 0.9 0.999 0.0001 --bs 100 --ne 300 --wu 50 --ap gaussian --ent convolutional --ehd "3 64 4 2 0, 64 128 4 2 0, 128 256 4 2 0, 1024 100" --dnt deconvolutional --dhd "50 1024, 256 128 4 2 0, 128 64 4 2 1, 64 3 4 2 0" --dur 1 --eur 1

# Experiment 3

# Experiment 4

# Experiment 5

# Experiment 6

# Experiment 7

# Experiment 8

# Experiment 9
