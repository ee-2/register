# Embeddings

The feature package *embeddings* takes document embeddings into account. This feature package is based on embeddings provided by [flair](https://github.com/flairNLP/flair) (Akbik, Alan; Bergmann, Tanja; Blythe, Duncan; Rasul, Kashif; Schweter, Stefan and Vollgraf, Roland. 2019: FLAIR: An easy-to-use framework for state-of-the-art NLP. NAACL 2019, 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations), 54-59.).

Chooseable parameters are:
* name : str
	- name of the feature package (defaults to *embeddings*)
* lang : str
	- ISO-code of language to use
* doc_embeddings : {'DocumentPoolEmbeddings', 'TransformerDocumentEmbeddings', 'SentenceTransformerDocumentEmbeddings'}
	- [document embeddings](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md) from flair, default is 'DocumentPoolEmbeddings'
	- choose from flair's DocumentPoolEmbeddings, TransformerDocumentEmbeddings, SentenceTransformerDocumentEmbeddings
* embeddings : dict, optional
	- token embeddings (can be stacked too), default loads Classic Word Embeddings for the chosen language  
	- choose from flair's (stacked) [word embeddings](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md)
* config : dict, optional
	- configuration for DocumentPoolEmbeddings, default keeps flair's defaults (fine\_tune\_mode = linear, pooling = first)
	- see [documentation](https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/DOCUMENT_POOL_EMBEDDINGS.md)


Configuration example:
````
    {
        "feature_package": "embeddings",
        "embeddings": {
                       "TransformerWordEmbeddings":{
                                                    "model": "bert-base-uncased",
                                                    "config": {
                                                              "layers": "all"
                                                              }
                                                   } 
                      }
    }}
````
