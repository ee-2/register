# Vectorizers
For some feature packages (see the [feature packages](1_basic_configurations.md/## Feature Packages) configuration) a vectorizer for building feature vectors can be set.
**register** implements a wrapper for the [sklearn](https://scikit-learn.org) vectorizers [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) or [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html).
Thus, you can decide to use the (relative) frequencies of the 1000 most common features, binary encoded features, TF-IDF features and so on. 
Set the *vectorizer* in the feature package configuration. Define the name of the chosen vectorizer in *name* and add available parameters for the method directly as specified by sklearn if necessary. 

Chooseable parameters are:
* name : {'CountVectorizer', 'TfidfVectorizer'}
	- name of sklearn vectorizer
* sklearn configuration parameters, optional
	- set configuration parameters for the sklearn vectorizer directly
	- see sklearn's documentation for possible configurations and set them here in the same way; defaults correspond to sklearn's defaults
	- parameters regarded to preprocessing cannot be set (text is already prepocessed)

## Configuration Examples

Use binary counts (1 for all non-zero counts) of 1000 most common features across corpora:
````
                "vectorizer": {
                               "name": "CountVectorizer",
							   "binary":true,
                               "max_features": 1000
                              }
````

Use frequencies of terms that appear at least in 2 documents (min_df=2) and not in more than 50% (max_df=0.5) of the documents:
````
                "vectorizer": {
                               "name": "CountVectorizer",
							   "max_df":0.5,
							   "min_df":2
                              }
````

Use TF-IDF values of all features:
````
                "vectorizer": {
                               "name": "TfidfVectorizer"
                              }
````
Use relative frequencies (sum of row values is 1):
````
                "vectorizer": {
                               "name": "TfidfVectorizer",
							   "use_idf":false,
							   "norm": "l1"
                              }
````
