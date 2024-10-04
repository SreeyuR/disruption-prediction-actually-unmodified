Author: Lucas Spangher.

# Plasma Disruption Prediction with GPT2 Transformers 

Disruption prediction for nuclear fusion plasmas using GPT2-style transformers.

This work aims to expand hybrid deep learning prediction for nuclear fusion plasma disruptions on CMOD, EAST, and Diii-D fusion reactors.

### A brief orientation: 
- Model_training/ has all of the files necessary for training
- Model_training/disruption_training_script.py serves as a main file for executing the training. Please try 

```python3 disruption_training_script.py```

for the default run, or

```bash master_sbatch_dispatch_script.sf```

if you'd like to launch jobs on a GPU. 
- requirements.txt should be up to date, but if not, then I used python3.8 and downloaded all packages during May 2023, and so creating a virtualenv with python3.8 and pip3 <package-name> should bring the correct version. 
- dvc should be used for ~2.2Gb of datafiles. This will require installing dvc and dvc[gdrive]:

```pip install dvc```
```pip install dvc\[gdrive\]``` if on zsh shell, or some other combination otherwise. 

### Projects:
- Include intentional disruptions (on hold till we can access new data)
- include device indicators and metaparameters (on hold)
- include control parameters and direct temperature of the inner wall prior to start (on hold)
- include time of day, month, and other proxy variables including 1st, 2nd, shot of day (on hold)
- train model on entire shot, check out attention weights (on hold)
- transformer pretrained on state prediction (on hold)
- multi-task transformer trained on both, and confinement regimes! (future)
- test a TimeSeriesTransformer in addition to a GPT2SequenceClassification transformer. (future)
- write tests for the code (future). 
- fine-tuning on individual reactors (on hold)
- loss based on disruption severity (on hold)
- augment original dataset with NaN-style window sampling (future). 

### Completed projects or steps:
- reconstructed NaN values w masked auto-encoder (Work is done and we've justified that this is not important.)
- modify and prepare the script for large-scale hyperparameter tuning (almost there!) 
- modify scaling such that we perform robust scaling on the training set, and use the scale parameters on the test set. (Done.)
- standardize sampling rate (Done -- linear interpoliation. Will is working on S4 interpolation.)
- implementation of sequential prediction loss (done! testing for class imbalance.)
