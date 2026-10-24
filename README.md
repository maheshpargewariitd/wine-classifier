\# Wine Classifier



This project uses machine learning to predict a wine's class from its

measured chemical features.



\## Model



The model uses a scikit-learn pipeline with feature scaling and a KNN

classifier. Cross-validation helps select the number of neighbors.



\## Run it



Activate the `campusx` environment, then run these commands from the

project folder:



```bat

python src\\train.py

python src\\predict.py

