# ont_m6a
m6A calling from ONT data. Repository for UW GS hackathon


## ML conda env

```
conda env create -f envs/ml_env.yml
```

## Making ML dataset (for new datasets)

```
python m6a_ont_data.py --positive_path ../data/HG002_2_00.npz --negative_path ../data/HG002_3_00.npz
```

## Running supervised CNN network

```
python m6a_supervised_cnn.py --config_file ../config.yml
```

## Running semi-supervised CNN network (after supervised run, as it needs supervised CNN model to initialize)

```
python m6a_semi_supervised_cnn.py --config_file ../config.yml
```