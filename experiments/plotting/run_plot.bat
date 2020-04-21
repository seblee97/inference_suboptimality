
python plot_from_df.py -save_path . -plot_keys plot_config.json -exp_name test -rolling_mean 50

python plot_from_df.py -save_path . -plot_keys plot_config.json -exp_name test

python plot_from_df.py -save_path . -plot_keys plot_config.json -exp_name test -zoom 250

python plot_from_df.py -save_path . -plot_keys plot_config.json -exp_name test -rolling_mean 50 -zoom 250 -compression 5

python plot_from_df.py -save_path . -csv "exp2_MNIST_FFG_AF.csv" -plot_keys plot_config.json -exp_name lo -lo True