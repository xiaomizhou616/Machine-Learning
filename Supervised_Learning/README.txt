Neural Network

Prerequisites
- Python 2.7
- pytorch
- numpy
- matplotlib

To install pytorch: pip install torch torchvision

1. Protein solubility
cd ./protein_solubility

To tune the number of layers and neurons through cross validation:
python tune_para.py --nLayers 0 --nNeurons 20 (0 hidden layers and 20 neurons per layer)

To train and validate the model with a propotion of data:
python exp_data_ratio.py --percentage 0.5 (train with 50% training data)

To plot the curve of accuracy versus epochs:
python acc_vs_epochs.py

2. Bank data
cd ./bank

To tune the number of layers and neurons through cross validation:
python tune_para.py --nLayers 3 --nNeurons 10 (3 hidden layers and 10 neurons per layer)

To train and validate the model with a propotion of data:
python exp_data_ratio.py --percentage 0.5 (train with 50% training data)

To plot the curve of accuracy versus epochs:
python acc_vs_epochs.py


Decision Tree, SVM, Boosting, KNN

1. Set the working directory as the folder "xhan306", Run code "project_1.R" in R version 3.5.1

2. For tables in the report, change the parameters and record them in the table. For figures in the report, record results in "csv" files and them plot them in excel. 

