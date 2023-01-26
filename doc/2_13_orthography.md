# Orthography and Token-Token Ratios

The feature package *orthography* contains mostly orthography and mediality related features. 

Chooseable parameters are:
* name : str, optional
	- name of the feature package, if not set defaults to *orthography*
* features : list, default=['prop_upper','prop_lower','prop_title','prop_punctuation','prop_emo']
	- *prop_upper*: proportion of upper cased words
	- *prop_lower*: proportion of lower cased words
	- *prop_title*: proportion of words with title case
	- *prop_punctuation*: proportion of punctuation
	- *prop_emo*: proportion of emojis and emoticons
    - *prop_contractions*: proportion of contractions (only contractions with apostrophe get counted)
* scaler : dict or bool, default={"name": "MinMaxScaler}
	- scaler to scale features with
	- set to *false* if you do not want to scale the features
	- choose from scalers provided by sklearn (e.g. [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) or [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler))
	- see [scaling](6_scaling.md) for further configurations

Configuration example:
````
        {
            "feature_package": "orthography",
            "features": ["prop_lower","prop_punctuation"]
        }
````
