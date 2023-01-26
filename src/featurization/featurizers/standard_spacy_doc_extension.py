from spacy.tokens import Doc
from spacy.language import Language
from statistics import mean

@Language.factory("StandardDocExtension")
def standard_spacy_doc_ext(nlp, name, attrs=('n_sents',
                                             'n_tokens',
                                             'n_tokens_punct',
                                             'n_words', 
                                             'n_types_token',
                                             'avg_token_len_chars',
                                             'avg_sent_len_tokens',
                                             'avg_sent_len_chars')):
    """Get standard extensions for spacy Doc object 
    
    Loaded in all use cases. Needed for multiple feature packages.

    Parameters
    ----------        
    nlp : spaCy Language object
    name: str
        name of spaCy extension (StandardDocExtension)
    attrs : tuple, default=('n_sents','n_tokens','n_tokens_punct', 'n_words', n_types_token','avg_token_len_chars','avg_sent_len_tokens','avg_sent_len_chars')
        attributes to set on Doc
                
    Returns
    -------
    StandardSpacyDocExt object
    """
    return StandardSpacyDocExt(attrs)  


class StandardSpacyDocExt():
    """SpaCy Extension for basic Doc attributes

    Parameters
    ----------        
    nlp : spaCy Language object
    attrs : tuple
        attributes to set on Doc    
    """     
    def __init__(self, attrs):
        self._n_sents, self._n_tokens, self._n_tokens_punct, self._n_words, self._n_types_token, self._avg_token_len_chars, self._avg_sent_len_tokens, self._avg_sent_len_chars = attrs
        if not Doc.has_extension(self._n_sents):
            Doc.set_extension(self._n_sents, getter=self._get_n_sents)
        if not Doc.has_extension(self._n_tokens):
            Doc.set_extension(self._n_tokens, getter=self._get_n_tokens)
        if not Doc.has_extension(self._n_tokens_punct):
            Doc.set_extension(self._n_tokens_punct, getter=self._get_n_tokens_punct)
        if not Doc.has_extension(self._n_words):
            Doc.set_extension(self._n_words, getter=self._get_n_words)
        if not Doc.has_extension(self._avg_sent_len_chars):
            Doc.set_extension(self._avg_sent_len_chars, getter=self._get_avg_sent_len_chars)
        if not Doc.has_extension(self._n_types_token):
            Doc.set_extension(self._n_types_token, getter=self._get_n_types_token)
        if not Doc.has_extension(self._avg_token_len_chars):
            Doc.set_extension(self._avg_token_len_chars, getter=self._get_avg_token_len_chars)
        if not Doc.has_extension(self._avg_sent_len_tokens):
            Doc.set_extension(self._avg_sent_len_tokens, getter=self._get_avg_sent_len_tokens)
        if not Doc.has_extension(self._avg_sent_len_chars):
            Doc.set_extension(self._avg_sent_len_chars, getter=self._get_avg_sent_len_chars)
        """
        extend here
        """
        
    def __call__(self, doc):
        """Process doc and set attributes

        Parameters
        ----------
        doc : spaCy doc

        Returns
        -------
        doc : spaCy doc                                    
        """
        return doc
    
    def _get_n_sents(self, doc):
        """Get the number of sentences in doc 
        
        Parameters
        ----------
        doc : spaCy doc

        Returns
        -------
        int 
        """
        return sum(1 for _ in doc.sents) 
        
    def _get_n_tokens(self, doc):
        """Get the number of tokens in doc 
        
        Parameters
        ----------
        doc : spaCy doc

        Returns
        -------
        int 
        """
        return len(doc)

    def _get_n_tokens_punct(self, doc):
        """Get the number of punctuation tokens in doc 
                
        Parameters
        ----------
        doc : spaCy doc

        Returns
        -------
        int 
        """
        return sum(1 for token in doc if token.is_punct)

    def _get_n_words(self, doc):
        """Get the number of tokens in doc with punctuation filtered 
                
        Parameters
        ----------
        doc : spaCy doc

        Returns
        -------
        int 
        """
        return sum(1 for token in doc if not token.is_punct)
    
    def _get_n_types_token(self, doc):
        """Get the number of types in doc 
                
        Parameters
        ----------
        doc : spaCy doc

        Returns
        -------
        int 
        """
        return len(set(token.text for token in doc))

    def _get_avg_token_len_chars(self, doc):
        """Get average length in characters of tokens in doc 
                
        Parameters
        ----------
        doc : spaCy doc

        Returns
        -------
        int or float
        """
        return mean([len(token.text) for token in doc])

    def _get_avg_sent_len_tokens(self, doc):
        """Get average length in tokens of sentences in doc 
                        
        Parameters
        ----------
        doc : spaCy doc

        Returns
        -------
        int or float
        """
        return mean([len(sent) for sent in doc.sents])
    
    def _get_avg_sent_len_chars(self, doc):
        """Get average length in characters of sentences in doc 
            
        Parameters
        ----------
        doc : spaCy doc

        Returns
        -------
        int or float
        """
        return mean([len(sent.text) for sent in doc.sents])