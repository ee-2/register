# Morphology
The feature package *morphology* includes morphological annotations provided by [spaCy](https://spacy.io/api/morphology#morphanalysis) and based on [Universal Dependencies](https://universaldependencies.org/format.html#morphological-annotation). 
It's possible to stack multiple different *morphology* feature packages, but be sure to name them differently (*name*). 

**Attention**: Only available for [spaCy language models](https://spacy.io/usage/models) with *morphologizer* for predicting morphological features. See spaCy's documentation for your language model.

Chooseable parameters are:
* name : str, optional
	- name of the feature package, if not set defaults to *morphology*
	- if you want to stack multiple *morphology* feature packages, set an unique name to avoid conflicts
* morph_tags : list, optional
	-  morphological tags to take into account, default takes all morph_tags (defined under [UD](https://universaldependencies.org/u/feat/index.html)):
		['Abbr','AdpType','AdvType','Animacy','Aspect','Case','Clusivity','ConjType',
         'Definite','Degree','Deixis','DeixisRef','Echo','Evident','Foreign','Gender',
         'Gender[dat]','Gender[erg]','Gender[obj]','Gender[psor]','Gender[subj]','Hyph',
         'Mood','NameType','NounClass','NounType','NumForm','NumType','NumValue',
         'Number','Number[abs]','Number[dat]','Number[erg]','Number[obj]','Number[psed]',
         'Number[psor]','Number[subj]','PartType','Person','Person[abs]','Person[dat]',
         'Person[erg]','Person[obj]','Person[psor]','Person[subj]','Polarity','Polite',
         'Polite[abs]','Polite[dat]','Polite[erg]','Poss','PrepCase','PronType','PunctSide',
         'PunctType','Reflex','Style','Subcat','Tense','Typo','VerbForm','VerbType','Voice']
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
                "feature_package": "morphology",
                "name": "CNG",
                "morph_tags": ["Case","Number","Gender"],
                "vectorizer":    {
                                    "name": "TfidfVectorizer"
                                 },
                "scaler":        {
                                    "name": "Normalizer"
                                 }
            },
            {
                "feature_package": "morphology",
                "name": "Tense"
                "morph_tags": ["Person"]            
            }
````
