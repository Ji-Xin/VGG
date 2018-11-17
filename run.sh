export OMP_NUM_THREADS=8
mkdir -p log
mkdir -p figs
mkdir -p output

# preprocess data
cd data
python preprocess.py
cd ..

# raw training and testing
python -u src/run.py --data_dir data --output_dir output --fig_dir figs --marker raw 1>log/raw 2>log/raw2
python -u src/run.py --data_dir data --output_dir output --fig_dir figs --marker raw --test

# l2 training and testing
python -u src/run.py --data_dir data --output_dir output --fig_dir figs --l2_reg --marker l2 1>log/l2 2>log/l22
python -u src/run.py --data_dir data --output_dir output --fig_dir figs --marker l2 --test

# augmented training and testing
python -u src/run.py --data_dir data --output_dir output --fig_dir figs --marker aug --aug --l2_reg 1>log/aug 2>log/aug2 
python -u src/run.py --data_dir data --output_dir output --fig_dir figs --marker aug --test
