# Named Entities
The feature package *named\_entities* retrieves named entities (determined with spaCy) for stylistic analysis. The entity types can be specified. Define the way of counting these entities via *vectorizer*.

**Attention**: Only available for [spaCy language models](https://spacy.io/usage/models) with *ner* for named entity recognition. See spaCy's documentation for your language model.

Chooseable parameters are:
* name : str, optional
	- name of the feature package, if not set defaults to *named\_entities*
* entities : list, optional
    - entities to take into account (e.g. ['ORG','LOC']), default all entities specified for language
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
            "feature_package":"named_entities",
            "vectorizer":    {
                                "name":"CountVectorizer",
                                "binary": true
                             }
        }
````
