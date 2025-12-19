# Instructions
## Obtaining the data
This repository contains a script which uses random forest to attemp solving the ["Store Sales Time Series"](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition from Kaggle. The data csv files can be obtained [here](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).    
Alternately, they can be obtained by running
```bash
kaggle competitions download -p var -c store-sales-time-series-forecasting
```
and then
```bash
unzip var/store-sales-time-series-forecasting.zip -d var/
```
## Training
The script uses DVC to run the training and inference steps sequentially. Once the conda environment has been created, in the repository's root directory, run
```bash
dvc repro
```
to test running a full pipeline. The training step will obtain its hyper-parameters from `utils/config.yml`. The metrics will be stored in `var/training\_metrics.json`. The model artifacts will then be stored in `var/model.pkl`.  
During the inference step, the artifacts are loaded into the model, and the inference results are saved in the `submission.csv` file.  
## Modifying parameter
Parameters can be changed while executing a pipeline using the -S argument of `dvc experiments run` with file path, followed by a colon and then followed by the path to the parameter, with periods to traverse the dictionary structure. E.g. to change the number of estimators to 50, run:
```bash
dvc experiments run -S utils/config.yml:random_forest_parameters.n_estimators=50
```
The metrics and parameters of completed experiments can be viewed in the `less` pager by executing:
```bash
dvc experiments show
```
## Unit tests
The unit tests are in `utils/tests.py`.  
They can be run by executing
```bash
pytest -v utils/tests.py
```
