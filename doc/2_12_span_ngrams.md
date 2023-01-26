# Span N-Grams
The feature package *span\_ngrams* lets you take different [spaCy span objects](https://spacy.io/api/span) into account or specify n-gram types. It's possible to stack multiple different span n-gram types by declaring multiple *span\_ngram* feature packages, but be sure to name them differently (*name*). 

**Attention**: Not every attribute for spaCy Span is covered by all [spaCy language models](https://spacy.io/usage/models). See spaCy's documentation for your language model.

Chooseable parameters for *span\_ngrams* are:
* name : str, optional
	- name of the feature package, if not set defaults to *span\_ngrams*
	- if you want to stack multiple *span\_ngrams* feature packages, set an unique name to avoid conflicts
* n : int, default=1
	- choose the n for the n-grams, i.e. 2 for bigrams
* spacy\_attrib : str, default="ents"
	- spacy attribute to get ngrams from, i.e. ents, noun_chunks etc.
	- see [spaCy's Doc docu](https://spacy.io/api/doc) for possible attributes
* exclude_spacy_attribs : list, default=[]
	- define list with spacy attributes which should not be indcluded, e.g. punctuation or stop words
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
            "feature_package":"span_ngrams",
            "name": "named_entities"
        }
````
