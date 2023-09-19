# ont_m6a
m6A calling from ONT data. Repository for UW GS hackathon

## Stored data for modelling

```
/net/gs/vol4/shared/public/hackathon_2023/Stergachis_lab/data/ml_data/HG002_2_3_00_train.npz
/net/gs/vol4/shared/public/hackathon_2023/Stergachis_lab/data/ml_data/HG002_2_3_00_val.npz
/net/gs/vol4/shared/public/hackathon_2023/Stergachis_lab/data/ml_data/HG002_2_3_00_test.npz
```
You can get on the GS cluster and run the following snippet to copy data over 

```
cd data
sh softlink_data.sh
```

## ML conda env

```
conda create --name ont_m6a

conda activate ont_m6a

conda install -c anaconda python=3.8

conda install -c conda-forge matplotlib

#conda install -c conda-forge numpy

conda install -c anaconda scipy

conda install -c anaconda scikit-learn

module load cuda/11.7.1

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Everytime you enter a compute node using qlogin, you will need to activate the conda env

```
conda activate ont_m6a
```

## Making ML dataset (for new datasets)

```
python m6a_ont_data.py --positive_path ../data/HG002_2_00.npz --negative_path ../data/HG002_3_00.npz --save_path ../data/HG002_2_3_00
```

## Running supervised CNN network

```
python m6a_supervised_cnn.py --config_file ../config.yml
```

## Running semi-supervised CNN network (after supervised run, as it needs supervised CNN model to initialize)

```
python m6a_semi_supervised_cnn.py --config_file ../config.yml
```
