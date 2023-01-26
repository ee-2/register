# Extending **register**

## Build your own Feature Package

To build your own feature package you have to define a module under *featurizers* containing a featurization class with an attribute called *config* (set on initialization).
The *config* attribute has to include:
* name : str
	- name of the feature package
* scaler : dict or bool
	- configuration for scaler to scale features with	
* vectorizer : dict, optional
	- if your feature package returns a dict with lists or another special configuration set configuration for vectorizer here too
* features : list, optional
	- if the features to use can be set for running **register** and feature values are scalar, be sure to set them here for constituency reasons (comparison of different languages, loading of pretrained models etc.)

If your feature package includes other configuration parameters which need to be consistent throughout different runs set them in *config* too. But keep in mind not to set path variables etc. here, because they can differ between languages etc.

Your featurizer class also needs a function called "featurize_doc" which takes a spaCy doc as parameter and returns a dictionary with the calculated values for for each feature or a dictionary with the feature package name as key and one value. Currently the values can either be scalar or list. For other options you have to build your own vectorizer (see bottom of this page). 

Implementation example:
````
class ExampleFeaturizer():
    """Example Featurizer
    
    Parameters
    ----------
    name : str
        name of the feature package
    features : list, default=['prop_upper_token', 'nr_sents']
        prop_punct: proportion of punctuation
        nr_sents: number of sentences in document
    scaler : dict or bool, default={"name": "MinMaxScaler}
        scaler to scale features with
        set to 'false' if you do not want to scale the features

    Attributes
    ----------
    config : dict
        the configurations for the feature package,
        save configuration parameters and defaults in config 
        name is obligatory for all feature packages
        set configuration for scaler here
        be sure to set features (if choosable) for constituency reasons
		do not set path variables here
        if your feature package returns a dict with lists or a special configuration set configuration for vectorizer here too
        if your feature package returns a dict with values for chosen features, dict vectorizer is used which can't be changed and is not set in config

    Notes
    -----    
    define in config file under feature_packages via:
        {
            "feature_package":"example",
            "name": "exp1"
        }
    """
    
    def __init__(self, name, 
                 features=['prop_punct', 'nr_sents'],
                 scaler={'name':'MinMaxScaler'}):
        self.config={'name':name,
                     'features':features,
                     'scaler':scaler}

    def featurize_doc(self, doc):
        """Get features for doc 

        Parameters
        ----------
        doc : spaCy Doc
        
        Returns
        -------
        dict
            dictionary with feature names as keys and feature values as values ({"prop_punct":0.21, "nr_sents":18})
        """
        data = {}
        data['prop_punct'] = doc._.n_tokens_punct / doc._.n_tokens if doc._.n_tokens else 0
        data['nr_sents'] = doc._.n_sents
        return {k:v for k,v in data.items() if k in self.config['features']}

````

Extend the function *_set_language_independent_feature_modules* for language independent feature packages in [featurizers/__init__.py](../src/featurization/featurizers/__init__.py) or *_set_language_dependent_feature_modules* for language dependent feature packages in the language dependend featurizers (e.g., [en/__init__.py](../src/featurization/featurizers/lang/en/__init__.py)) to initialize your feature package too.

Initialization of the example feature package:
````
if 'example' in feature_packages:
    from .example import ExampleFeaturizer
    for feature_package, feature_package_config in feature_packages['example'].items():
        self.feature_modules[feature_package] = ExampleFeaturizer(**feature_package_config)

````

## Build your own Machine Learning Model

Under *models* and the type of your model (classification, linear regression or clustering), open a file and define a class called the model type followed by and underline and a parameter which is then set via *lib* (e.g., 'classification\_sklearn'). The super classes [classification.py](../src/models/classification/classification.py), [linearRegression.py](../src/models/linearRegression/linearRegression.py) and [clustering.py](../src/models/clustering/clustering.py) define necessary functions for the models, you can either inherit them from the super classes or define/overwrite them by yourself. For using the standard functions of the super classes your model class must have the attributes *model*, which contains the machine learning model itself, and *_scalers*, which is a dictionary including the scalers per each feature package. See the '*_tf_keras' and *_sklearn* implementations for examples.


## Build your own Vectorizer

To build your own vectorizer extend [vectorizers.py](../src/featurization/vectorizers.py) with your vectorizer class. This class has to implement a function *vectorize*, which takes x and returns it in a vectorized format as a 2-dimensional [numpy array](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html). If the vectorizer needs to be reused for when loading a pretrained model the vectorizer class must have a vectorizer attribute which includes the actual vectorizer model.

Make sure to instantiate your vectorizer correctly in the functions *_vectorize* and *_vectorize_pretrained* in [featurization/__init__.py](../src/featurization/__init__.py).

Example vectorizer (stacks embeddings):
````
class EmbeddingsVectorizer():
    """Vectorize embeddings 
    """
    def vectorize(self, x):
        """Stack embeddings vertically
        
        Parameters
        ----------
        x : list of ndarray
        
        Returns
        -------
        ndarray
        """
        return np.vstack(x)
````

