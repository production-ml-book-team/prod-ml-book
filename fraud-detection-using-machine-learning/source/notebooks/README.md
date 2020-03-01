# Setup AWS credentials 
Run the following command in a terminal that is in the same enironment as the notebook interface.  

> aws configure 

Enter in the Access Key from your AWS account and the Secret Key when prompted. Choose the us-west-1 region and json output

# Setup DVC 
Ensure that DVS will run in the same environment as the AWS credentials which are inputted above. DVC and AWS S3 will work if the credentials are available within the environment.  


# Example: Simple Pipeline with DVC 

Find the original reference here: https://dvc.org/doc/get-started/example-pipeline

The pipeline we will build for a simple consists of the following steps and also produces performance metrics of the trained model. 

1. loading data
2. training
3. evaluation

Instructions 

## 1. loading data
> dvc run -f load.dvc -d config/load.json -o data python code/load.py

Output
```
Running command:
        python code/load.py
Computing md5 for a large directory data/2. This is only done once.
...
```
> git status
        .gitignore
        load.dvc
> cat .gitignore data

> git add .gitignore load.dvc
> git commit -m "init load stage"

# 2. training
> dvc run -f train.dvc -d data -d config/train.json -o model/model.h5 python code/train.py

Output
```
...
```

# 3. evaluation
> dvc run -f evaluate.dvc -d model/model.h5 -M model/metrics.json python code/evaluate.py

Output
```
...
```

# Reproducing the entire pipeline
> dvc repro evaluate.dvc

Output
```...
Stage 'load.dvc' didnt change.
Stage 'train.dvc' didnt change.
Stage 'evaluate.dvc' didnt change.
Pipeline is up to date. Nothing to reproduce.
```
