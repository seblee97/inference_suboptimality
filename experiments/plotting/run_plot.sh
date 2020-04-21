# run with rolling_mean
python plot_from_df.py -save_path . -plot_keys plot_config.json -exp_name test -rolling_mean 50
# run without rolling mean
python plot_from_df.py -save_path . -plot_keys plot_config.json -exp_name test
#run with a compression before 250 epochs to default 10% of scale
python plot_from_df.py -save_path . -plot_keys plot_config.json -exp_name test -zoom 250
#run with a compression before 250 epochs to 5% of scale, with rolling mean
python plot_from_df.py -save_path . -plot_keys plot_config.json -exp_name test -rolling_mean 50 -zoom 250 -compression 5
#run for a local optimisation graph
python plot_from_df.py -save_path . -csv "exp2_MNIST_FFG_AF.csv" -plot_keys plot_config.json -exp_name lo -lo True