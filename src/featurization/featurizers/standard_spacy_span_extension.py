from spacy.tokens import Span
from spacy.language import Language

@Language.factory("StandardSpanExtension")
def standard_spacy_span_ext(nlp, name, attrs=('n_tokens', 'n_tokens_punct', 'n_words', 'n_sents')):
    """Get standard extensions for spacy Span object 
    
    Loaded in all use cases. Needed for multiple feature packages.

    Parameters
    ----------        
    nlp : spaCy Language object
    name: str
        name of spaCy extension (StandardSpanExtension)
    attrs : tuple, default=('n_tokens', 'n_tokens_punct', 'n_words', 'n_sents')
        attributes to set on Span
        number of sents is only set for consistency
                
    Returns
    -------
    StandardSpacySpanExt object
    """
    return StandardSpacySpanExt(attrs)  


class StandardSpacySpanExt():
    """SpaCy Extension for basic Span attributes

    Parameters
    ----------        
    nlp : spaCy Language object
    attrs : tuple
        attributes to set on Span    
    """ 
    def __init__(self, attrs):
        self._n_tokens, self._n_tokens_punct, self._n_words, self._n_sents = attrs
        if not Span.has_extension(self._n_tokens):
            Span.set_extension(self._n_tokens, getter=self._get_n_tokens)
        if not Span.has_extension(self._n_tokens_punct):
            Span.set_extension(self._n_tokens_punct, getter=self._get_n_tokens_punct)
        if not Span.has_extension(self._n_words):
            Span.set_extension(self._n_words, getter=self._get_n_words)
        if not Span.has_extension(self._n_sents):
            Span.set_extension(self._n_sents, getter=self._get_n_sents)
        """
        extend here
        """
        
    def __call__(self,doc):
        """Process doc and set attributes

        Parameters
        ----------
        doc : spaCy doc

        Returns
        -------
        doc : spaCy doc                                    
        """
        return doc
        
    def _get_n_tokens(self, span):
        """Get the number of tokens in span 

        Parameters
        ----------
        span : spaCy span

        Returns
        -------
        int 
        """
        return len(span)
    
    def _get_n_tokens_punct(self, span):
        """Get the number of punctuation tokens in span 

        Parameters
        ----------
        doc : spaCy span

        Returns
        -------
        int  
        """
        return sum(1 for token in span if token.is_punct)

    def _get_n_words(self, span):
        """Get the number of tokens in span with punctuation filtered 
                
        Parameters
        ----------
        span : spaCy span

        Returns
        -------
        int 
        """
        return sum(1 for token in span if not token.is_punct)

    def _get_n_sents(self, span):
        """Return 1 for consistency 

        To handle doc and span in the same way return 1 for the number of sentences in a sentence (span)
                
        Parameters
        ----------
        span : spaCy span

        Returns
        -------
        int 
            1
        """
        return 1
    