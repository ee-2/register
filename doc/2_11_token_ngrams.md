# Token N-Grams

The feature package *token\_ngrams* lets you include all attributes spaCy defines for its [Token](https://spacy.io/api/token). It's possible to stack multiple different token n-gram types by declaring multiple *token\_ngram* feature packages, but be sure to name them differently (*name*). 

**Attention**: Not every attribute for spaCy Token is covered by all [spaCy language models](https://spacy.io/usage/models). See spaCy's documentation for your language model.

Chooseable parameters for *token\_ngrams* are:
* name : str, optional
	- name of the feature package, if not set defaults to *token\_ngrams*
	- if you want to stack multiple *token\_ngrams* feature packages, set an unique name to avoid conflicts
* n : int, default=1
	- choose the n for the n-grams, i.e. 2 for bigrams
* spacy\_attrib : str, default="lemma\_"
	- spacy attribute to get ngrams from, i.e. text, lemma, POS-tag etc.
	- see [spaCy's Token docu](https://spacy.io/api/token) for possible attributes
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


Configuration example for bigrams of tokens as well as binary counts of unigrams of lemmas excluding stopwords.
````
            {
                "feature_package": "token_ngrams",
                "name": "token",
                "spacy_attrib":"text",
                "n" : 2
            },
            {
                "feature_package": "token_ngrams",
                "name": "lemma", 
                "exclude_spacy_attribs":["is_stop"],
                "vectorizer":    {
                                    "name":"CountVectorizer",
                                    "binary": true
                                 },
                "scaler": false
            }
````



Configuration example for unigrams of part-of-speech tags based on [Universal POS tags](https://universaldependencies.org/docs/u/pos/) provided by spaCy. 
**Attention**: Only available for [spaCy language models](https://spacy.io/usage/models) with *tagger* for pos tagging. See spaCy's documentation for your language model.
````
            {
                "feature_package": "token_ngrams",
                "spacy_attrib":"pos_"
            }
````


