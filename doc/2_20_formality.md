# Formality (German)

The feature package *formality* builds on formality-induced lexica and for now is only available for German. It is based on *I-ForGer* (Eder, Elisabeth; Krieg-Holz Ulrike and Hahn, Udo. 2021: Acquiring a Formality-Informed Lexical Resource for Style Analysis. Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, April 2021, Kyiv, Ukraine, Online, 2028--2041).

Chooseable parameters are:
* name : str, optional
	- name of the feature package, if not set defaults to *formality*
* level : {'document','token','sentence'}
	- level to calculate emotion on, default is *document*
	- for *document* and *sentence* the score is determined via the average of the included token scores
* path : : str, optional
	- path to I-ForGer lexicon (or similar lexicon)
	- default loads the lexicon from folder lang\_data/formality
* vectorizer : dict, default={"name":"TfidfVectorizer", "use_idf":false, "norm":"l1"}
	- vectorizer to vectorize features with
	- only applies if level is *token* or *sentence* , for level 'document' no vectorizer is taken
	- choose from [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) or [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) from sklearn
	- see [vectorizers](5_vectorizers.md) for further configurations
* scaler : dict or bool or None, optional
	- scaler to scale features with
	- set to 'false' if you do not want to scale the features
	- choose from scalers provided by sklearn (e.g. [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) or [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler))
	- default takes StandardScaler ({'name':'StandardScaler'}) for level *token* or *sentence* and for level *document* MinMaxScaler ({'name':'MinMaxScaler'})
	- see [scaling](6_scaling.md) for further configurations


Configuration example:
````
        {
            "feature_package": "formality",
            "name": "iforger",
			"level": "token"
        }
````
