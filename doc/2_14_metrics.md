# Metrics

The feature package *metrics* provides various metrics related to readability, formality, syntax, etc. 

Chooseable parameters for *metrics* are:
* name : str, optional
	- name of the feature package, if not set defaults to *metrics*
	- if you want to stack multiple *metrics* feature packages, set an unique name to avoid conflicts
* features : list, default=['avg_token_length', 'avg_sentence_length_tokens', 'avg_sentence_length_characters','avg_dependency_height', 'flesch_reading_ease', 'heylighen_f']
	- *avg\_token\_length*: average length of tokens
	- *avg\_sentence\_length\_tokens*: average length of sentences in tokens
	- *avg\_sentence\_length\_characters*: average length of sentences in characters
	- *avg\_constituency\_height*: average height of constituency trees of sentences, normalized by sentence length (constituency calculation via [benepar](https://github.com/nikitakit/self-attentive-parser))
	- *avg\_dependency\_height*: average height of dependency trees of sentences, normalized by sentence length
	- *flesch\_kincaid\_grade*: flesch-kincaid grade level (J. Peter Kincaid, Robert P. Fishburne Jr., Richard L. Rogers, and Brad S. Chissom. 1975: Derivation of new readability formulas (automated readability index, fog count and Flesch reading ease formula) for navy enlisted personnel. Technical report, DTIC Document.)
	- *flesch\_reading\_ease*: flesch reading ease (Flesch, Rudolf. 1948: A new readability yardstick. Journal of Applied Psychology 32:221-33.)
	- *heylighen\_f*: formality score as defined by Heylighen, Francis and Dewaele Jean-Marc. 1999: Formality of language: definition, measurement and behavioral determinants. Technical report, Center ”Leo Apostel”, Free University of Brussels.

* scaler : dict or bool, default={"name": "MinMaxScaler}
	- scaler to scale features with
	- set to *false* if you do not want to scale the features
	- choose from scalers provided by sklearn (e.g. [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) or [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler))
	- see [scaling](6_scaling.md) for further configurations


Configuration example to stack different grammars (be sure to name them differently (*name*)):
````
        {
            "feature_package":"metrics",
            "features": ["avg_dependency_height",
                        "heylighen_f"]
        }
````

For German *metrics* includes some further features.
The parameter list for *features* is extended by:

* features : list, default=['avg_token_length', 'avg_sentence_length_tokens', 'avg_sentence_length_characters','avg_dependency_height', 'flesch_reading_ease', 'heylighen_f']
	- *prop\_hedge\_phrases*: proportion of hedge words and phrases
	- *prop\_first\_prs\_pron*: proportion of first person pronouns
	- *prop\_third\_prs\_pron*: proportion of third person pronouns
