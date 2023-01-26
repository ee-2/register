# Character N-Grams

The feature package *character\_ngrams* lets you include character n-grams. It's possible to stack multiple different character n-gram types by declaring multiple *character\_ngram* feature packages, but be sure to name them differently (*name*). 

Chooseable parameters are:
* name : str, optional
	- name of the feature package, if not set defaults to *character\_ngrams*
	- if you want to stack multiple *character\_ngrams* feature packages, set an unique name to avoid conflicts
* n : int, default=3
	- choose the n for the n-grams, i.e. 2 for bigrams
* cross\_token\_boundaries : bool, default=True
	- whether to generate n-grams crossing token boundaries, if false tokens will be padded with whitespace to track start and end
* vectorizer : dict, default={"name":"CountVectorizer", "max_features":1000} 
	- vectorizer to vectorize features with
	- choose from [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) or [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) from sklearn
	- see [vectorizers](5_vectorizers.md) for further configurations
* scaler : dict or bool, default={"name": "StandardScaler}
	- scaler to scale features with
	- set to *false* if you do not want to scale the features
	- choose from scalers provided by sklearn (e.g. [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) or [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler))
	- see [scaling](6_scaling.md) for further configurations


Configuration example:
````
            {
                "feature_package": "character_ngrams",
            	"exclude_space":false
            }
````
