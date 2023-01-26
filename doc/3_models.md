# Models

**register** allows compare different machine learning models in one run. 
Just define them under *models*, where *model* names the model and *lib* defines the library to use.
Depending on the mode you choose there are different options for machine learning models. 



## Clustering

If the chosen mode is *cluster*, **register** provides a wrapper for the [sklearn](https://scikit-learn.org) clustering algorithms in the [sklearn.cluster](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster) module. As a default Agglomerative Clustering with its default parameters is performed.
To use different models or a different configuration, first set *lib* to *sklearn* and define the model under *model*. Then add other configuration parameters for the sklearn models as explained in the sklearn user guide (e.g. for [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) set *n_clusters*). 

Parameters:
* lib : str, default="sklearn"
	- library or file to get model from
	- currently only *sklearn* supported
* model : str, default="AgglomerativeClustering"
	- the exact name of the model from the [sklearn.cluster](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster) module
* sklearn configuration parameters, optional
	- set configuration parameters for the sklearn model directly
	- see sklearn's documentation for possible configurations and set them here in the same way; defaults correspond to sklearn's defaults


Configuration example:
````
	"models":[	

				{	"model": "Kmeans",
					"lib": "sklearn",
					"n_clusters": 2,
					"max_iter": 100
				}

	         ]
````

## Classification

For mode *train\_classifier* **register** employs [sklearn](https://scikit-learn.org) classification algorithms as well as models based on [Keras](https://keras.io/) in [TensorFlow](https://www.tensorflow.org/).

Parameters:
* lib : str, default="sklearn"
	- library or file to get model from
 	- currently *sklearn* and *tf_keras* supported
* model : str, default="SGDClassifier"
	- either the exact name of the model as defined by sklearn (lib is *sklearn*) or an arbitrary name if lib is *tf_keras*
* model specific parameters, optional
	- set configuration parameters for the chosen model directly 
	- see the libraries documentation for possible configurations and set them here in the same way; defaults correspond to the defaults


### sklearn

You can choose from sklearn models: [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), Support Vector Machines ([SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC), [NuSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC) and [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)), [linear classifiers with Stochastic Gradient Descent training](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier) (standard is SVM), [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier), [K-Nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html), [Naive Bayes](https://sklearn.org/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB) and [Multi-layer Perceptron Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html). 
Set *lib* to *sklearn* and define the chosen model under *model*. Assign any sklearn specific configuration parameters directly in the same way as explained in sklearn's documentation.

Configuration example:
````
	"models":[	

				{	"model": "LogisticRegression",
					"lib": "sklearn",
					"solver": "liblinear"
				}

	         ]
````

### tensorflow-keras

Set *model* to a name of your choice and *lib* to *tf_keras* to use models build with [Keras](https://keras.io/) in [TensorFlow](https://www.tensorflow.org/). Using *tf_keras* lets you define dimensions and numbers of hidden layers and dropout rate for input and hidden layers. If you need to set other parameters, feel free to [implement your own model](7_extending.md).

Model specific parameters:
* hidden_layer_dimensions : list, default=[128,64] 
	- list with dimensions of hidden layers, first entry is dimension of first hidden layer etc.
	- applies only if nr_hidden_layers is not set (0, default)
* nr_hidden_layers : int, default=0
	- if nr_hidden_layers specified, model with respective number of hidden layers is build
	- each dimension of a hidden layer is half the preceeding hidden layer (breaks if dim < 2)
* dropout_input : float, default=0.2
	- dropout rate to use on the input layer
* dropout_hidden : float, default=0.5
	- dropout rate to use on the hidden layers

Configuration example:
````
	"models":[	

				{	"model": "MyFancyNeuralNetwork",
					"lib": "tf_keras",
					"hidden_layer_dimensions": [
									256,
                                                  			128,
									 64
								   ]
				}

	         ]
````


### Linear Regression

For training a linear regression model **register** employs [sklearn](https://scikit-learn.org) linear regression algorithms as well as models based on [Keras](https://keras.io/) in [TensorFlow](https://www.tensorflow.org/).

Parameters:
* lib : str, default="sklearn"
	- library or file to get model from
 	- currently *sklearn* and *tf_keras* supported
* model : str, default="Ridge"
	- either the exact name of the model as defined by sklearn (lib is *sklearn*) or an arbitrary name if lib is *tf_keras*
* model specific parameters, optional
	- set configuration parameters for the chosen model directly 
	- see the libraries documentation for possible configurations and set them here in the same way; defaults correspond to the defaults


### sklearn

You can choose from sklearn models: [Multi-layer Perceptron Regressor](ttps://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) and models from [sklearn.linear_model](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model), e.g. [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression), [Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge) or [linear regressors with Stochastic Gradient Descent training](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor) (standard is linear regression).
Set *lib* to *sklearn* and define the chosen model under *model*. Assign any sklearn specific configuration parameters directly in the same way as explained in sklearn's documentation.

Configuration example:
````
	"models":[	

				{	"model": "SGDRegressor",
					"lib": "sklearn",
					"loss": "huber"
				}

	         ]
````

### tensorflow-keras

Set *model* to a name of your choice and *lib* to *tf_keras* to use models build with [Keras](https://keras.io/) in [TensorFlow](https://www.tensorflow.org/). Using *tf_keras* lets you define dimensions and numbers of hidden layers and dropout rate for input and hidden layers. If you need to set other parameters, feel free to [implement your own model](7_extending.md).

Model specific parameters:
* hidden_layer_dimensions : list, default=[128,64] 
	- list with dimensions of hidden layers, first entry is dimension of first hidden layer etc.
	- applies only if nr_hidden_layers is not set (0, default)
* nr_hidden_layers : int, default=0
	- if nr_hidden_layers specified, model with respective number of hidden layers is build
	- each dimension of a hidden layer is half the preceeding hidden layer (breaks if dim < 2)
* dropout_input : float, default=0.2
	- dropout rate to use on the input layer
* dropout_hidden : float, default=0.5
	- dropout rate to use on the hidden layers


Configuration example:
````
	"models":[	

				{	"model": "MyFancyNeuralNetwork",
					"lib": "tf_keras",
					"hidden_layer_dimensions": [
													256,
                                                  	128,
									 				64
								   ]
				}

	         ]
````

## Training a model (preparations)

### Targets to learn
For the mode *train_classifier* you can either specify a class or label associated with the corpus via the *class* parameter in the corpus configuration or point to a tsv file via *path\_targets*, which contains the text id and the associated class in the columns id and class. For an example of this tsv file see [path_targets](examples/classes.tsv). 

For the mode *train_linear_regressor* set *path\_targets* to a tsv file containing the text id and the associated score in the columns id and score. For an example of this tsv file see [path_targets](examples/scores.tsv). 

### Train and test sets
Further, you can define train and test sets via setting *set* in the corpus configuration to "train" or "test". If you use *set*, only one run with one fold will be processed.
You can also point to a tsv file containing the set for each text in a column for each fold (column starts with fold\*, multiple folds allowed). For an example see [path_train_test_splits](examples/train_test_splits.tsv).
If you don't set train and test sets the splits are determined randomly per default in a 10-fold cross-validation setting. You can change the number of folds by setting *cv_n_splits* in the first level of the configuration. 

Configuration example:
````
	"corpora":[	

					{
						"name": "corpus1",
						"path": "path_to_corpus1",
						"language": "de",
						"class": "class1"

					},
					{
						"name": "corpus2",
						"path": "path_to_corpus1",
						"language": "de",
						"class": "class2"

					}

	         ]
````

## Loading a pretrained model

For the modes *score* and *classify* you have to load a pretrained model. First set *base_dir* to the base folder the models, results and vectorizers of the previous run are in. In the model config set *lib* to the used library or file, set *model* to the chosen model's name and set *pretrained* to "modelName/modelName_fold_nr", i.e. the path to the pretrained model starting from the *base_dir* without the '_model' and '_model.pickle' endings.

Configuration example:
````
	"models":[	

				{	"model": "Ridge",
					"lib": "sklearn",
					"pretrained": "Ridge/Ridge_fold_1"
				}

	         ]
	"base_dir": "example_path"
````
