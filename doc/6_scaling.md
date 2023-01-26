# Scaling

For feature scaling **register** lets you choose a different scaling method for each feature package. **register** implements a wrapper for [sklearn](https://scikit-learn.org) [standardization and normalization](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) methods. Depending on the nature of the regarding feature package as a default [MinMaxScaler](ttps://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler) or [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) is used. If you want to use a different scaling option, set the *scaler* in the feature package configuration. Just define the name of the chosen sklearn module in *name* and add available parameters for the method directly as specified by sklearn if necessary. If you do not want to use a scaler for a feature package at all, set *scaler* to *false* (bool). 

Chooseable parameters are:
* name : str
	- name of sklearn scaling module
* sklearn configuration parameters, optional
	- set configuration parameters for the sklearn scaler directly
	- see sklearn's documentation for possible configurations and set them here in the same way; defaults correspond to sklearn's defaults


Configuration example (in the feature package configuration):
````
                "scaler": {
                               "name": "StandardScaler",
							   "with_mean":false
                          }
````

Configuration example (in the feature package configuration) for not scaling the features:
````
                "scaler": false
````
