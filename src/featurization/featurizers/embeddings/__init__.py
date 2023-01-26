from flair.data import Sentence
import importlib
from flair.embeddings import (
    BytePairEmbeddings,
    CharacterEmbeddings,
    ELMoEmbeddings,
    FastTextEmbeddings,
    FlairEmbeddings,
    OneHotEmbeddings,
    PooledFlairEmbeddings,
    StackedEmbeddings,
    TransformerWordEmbeddings,
    WordEmbeddings,
    DocumentPoolEmbeddings,
    DocumentRNNEmbeddings,
    SentenceTransformerDocumentEmbeddings,
    TransformerDocumentEmbeddings
)

class EmbeddingsFeaturizer():
    """Get document embeddings
    
    Based on embeddings provided by flair 
    (Akbik, Alan; Bergmann, Tanja; Blythe, Duncan; Rasul, Kashif; Schweter, Stefan and Vollgraf, Roland. 2019: FLAIR: An easy-to-use framework for state-of-the-art NLP. NAACL 2019, 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations), 54-59.)
    
    Parameters
    ----------
    name : str
        name of the feature package
    lang : str
        ISO-code of language to use
    doc_embeddings : {'DocumentPoolEmbeddings', 'TransformerDocumentEmbeddings', 'SentenceTransformerDocumentEmbeddings'}
        document embeddings from flair, default is 'DocumentPoolEmbeddings'
        choose from flair's DocumentPoolEmbeddings, TransformerDocumentEmbeddings, SentenceTransformerDocumentEmbeddings
        see: https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md
    embeddings : dict, optional
        token embeddings (can be stacked too), default loads Classic Word Embeddings for the chosen language  
        choose from flair's (stacked) word embeddings: https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md
    config : dict, optional
        configuration for DocumentPoolEmbeddings, default keeps flair's defaults (fine_tune_mode = linear, pooling = first)
        see: https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/DOCUMENT_POOL_EMBEDDINGS.md
        
    Attributes
    ----------
    config : dict
        the configurations for the feature package
    _name : str
        name of the feature package
    doc_embeddings : object
        document embeddings from flair
    
    Notes
    -----
    define in config file under feature_packages via:
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
    }    
    """
    def __init__(self, lang, name, doc_embeddings='DocumentPoolEmbeddings', embeddings={}, **config):
        self.config = {'name':name,
                       'doc_embeddings':doc_embeddings,
                       'embeddings': embeddings}
        self._name = name
        
        if doc_embeddings == 'DocumentPoolEmbeddings': # default: mean
            token_embeddings = []
            for embedding, embs_config in embeddings.items():
                if embedding == 'CharacterEmbedding':
                    token_embeddings.append(CharacterEmbeddings()) # for character embeddings  
                else:
                    token_embeddings.append(getattr(importlib.import_module('flair.embeddings'), embedding)(embs_config['model'],**embs_config.get('config',{})))                
            if not token_embeddings:
                token_embeddings.append(WordEmbeddings(lang))
                self.config['embeddings'] = {'WordEmbeddings': {'model':lang}}
            self.doc_embeddings = DocumentPoolEmbeddings([StackedEmbeddings(embeddings=token_embeddings)], **config)  
            if config:
                self.config['config'] = config
        elif doc_embeddings == 'TransformerDocumentEmbeddings':
            self.doc_embeddings = TransformerDocumentEmbeddings(embeddings['model'], **embeddings.get('config',{}))
        elif doc_embeddings == 'SentenceTransformerDocumentEmbeddings':
            self.doc_embeddings = SentenceTransformerDocumentEmbeddings(embeddings['model'], **embeddings.get('config',{}))

        
    def featurize_doc(self, doc):
        """Get Flair embedding
        
        Parameters
        ----------
        doc : spaCy Doc
        
        Returns
        -------
        dict
            dictionary with feature package name as key and document's vector (numpy.ndarray[ndim=1, dtype=float32]) as value
        """
        sentence = Sentence([token.text for token in doc])
        self.doc_embeddings.embed(sentence)
        return {self._name:sentence.embedding}
