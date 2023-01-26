# Emotion
The feature package *emotion* is based on *MEmoLon – The Multilingual Emotion Lexicon* (Buechel, Sven; Rücker, Susanna; Hahn, Udo. 2020: Learning and Evaluating Emotion Lexicons for 91 Languages. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, July 2020, Online, 1202--1217.).

To include these emotional annotations, please download the lexicon for your language and put it into [lang_data/emotion](../lang_data/emotion) keeping it's name as is or define the path to the lexicon via *path*.
You find lexica for 91 languages on [Zenodo](https://zenodo.org/record/3756607/files/MTL_grouped.zip?download=1) (2.4 GB). Further information is available under the [MEmoLon repository](https://github.com/JULIELab/MEmoLon).

Chooseable parameters are:
* name : str, optional
	- name of the feature package, if not set defaults to *emotion*
* features : list, default=['valence','arousal','dominance','joy','anger','sadness','fear','disgust']
	- emotion features to take into account, defaults to taking all features into account
* level : {'document','token','sentence'}
	- level to calculate emotion on, default is *document*
	- for *document* and *sentence* the score is determined via the average of the included token scores
* path : str or dict, optional
	- path to memolon lexicon
	- default loads the lexicon for a specific language (following its ISO code) from folder lang\_data/emotion
	- for multiple languages path is a dictionary with the languages' ISO code as keys and the respective paths as values
	- if only one language is used path can be a string containing the path
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
            "feature_package": "emotion"
        }
````

For English and German also sentiment annotations are offered based on [TextBlob](https://textblob.readthedocs.io/en/dev/) and [TextBlobDE](https://textblob-de.readthedocs.io/en/latest/). For English *polarity* and *subjectivity* can be calculated, for German only *polarity* is possible.


Thus, for English the parameter list for *features* is extended:
* features : list, default=['valence','arousal','dominance','joy','anger','sadness','fear','disgust','polarity','subjectivity']


And for German:
* features : list, default=['valence','arousal','dominance','joy','anger','sadness','fear','disgust','polarity']

Configuration example:
````
        {
            "feature_package": "emotion",
			"features":["polarity", "dominance"]
        }
````


