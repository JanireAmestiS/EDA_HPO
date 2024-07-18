#Estimation of Distribution Algorithms for Generative Adversarial and Convolutional Neural Network Hyper-Parameter Optimization

##EDA-CNN
Estimation of Bayesian Network Algorithm (EBNA) for Convolutional Neural Network (CNN) hyper-parameter optimization. 

###EDA-CNN
Evaluation of the individuals in a population is carried out with CNN training results.

###EDA-Surrogate
Random Forest surrogate model is used for individual evaluation. Surrogate model (CV_rf.pickle) approximates the fitness function of an individual given a configuration of hyper-parameters.

###EDA-CNN-Surrogate
Evaluation of the individuals is a combination of both evaluation methods: CNN training and surrogate model. 
Surrogate model evaluation is applied in all individuals. This evaluation helps to determine the promising solutions, which are finally evaluated with CNN training. 

Two methods are analysed in determining the criteria of selecting the promising solutions:
* Fixed threshold: it maintains a percentage during the generations.
* Dynamic percentage: it is reduced as generations go by being more selective in determining the promising solutions. 

###Multi-objective EDA-CNN
Objectives consist of:
* Minimizing the validation loss of the CNN training.
* Minimizing the required computational cost.

Non-dominating sorting and crowding distance methods are performed for the implementation. 

##EDA-DCGAN
Estimation of Bayesian Network Algorithm (EBNA) for Deep Convolutional Generative Adversarial Network (DCGAN) hyper-parameter optimization. 