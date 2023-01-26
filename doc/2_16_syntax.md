# Syntax

The feature package *syntax* takes frequencies of syntactical labels into account. 

Chooseable parameters for *syntax* are:
* name : str, optional
	- name of the feature package, if not set defaults to *syntax*
	- if you want to stack multiple *syntax* feature packages, set an unique name to avoid conflicts
* grammar : {'constituency', 'dependency'}
	- the grammar syntax trees are based on, default is 'dependency'
	- *dependency*: occurrences of POS of head, dependency relation and POS of child, including all combinations
	- *constituency*: occurrences of production rules (without lexicalizations)
* pos : {'tag', 'pos}
	- whether to use Universal Dependencies v2 POS tag set (pos, default) or a finer-grained POS tag set (tag)
	- for *dependency* only
* vectorizer : dict, default={"name":"CountVectorizer", "max_features":1000} 
	- vectorizer to vectorize features with
	- choose from [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) or [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) from sklearn
	- see [vectorizers](5_vectorizers.md) for further configurations
* scaler : dict or bool, default={"name": "StandardScaler}
	- scaler to scale features with
	- set to *false* if you do not want to scale the features
	- choose from scalers provided by sklearn (e.g. [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) or [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler))
	- see [scaling](6_scaling.md) for further configurations


Configuration example to stack different grammars (be sure to name them differently (*name*)):
````
        {
            "feature_package":"syntax",
            "name": "dependency",
            "pos": "tag"
        },
        {
            "feature_package":"syntax",
            "name": "constituency",
            "grammar": "constituency"
        }
````

Constituency-related features are based on [benepar](https://github.com/nikitakit/self-attentive-parser):
Kitaev, Nikita and Klein, Dan. 2018. Constituency Parsing with a Self-Attentive Encoder. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), July 2018, Melbourne, Australia, 2676--2686.
Kitaev, Nikita; Cao, Steven; and Klein, Dan. 2019. Multilingual Constituency Parsing with Self-Attention and Pre-Training. July 2019, Florence, Italy, 3499--3505.

If you choose constituency-related features, make sure to have [benepar](https://github.com/nikitakit/self-attentive-parser) loaded the respective model as described.
