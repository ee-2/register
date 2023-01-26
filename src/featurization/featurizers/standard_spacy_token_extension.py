from spacy.tokens import Token
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.lang.tokenizer_exceptions import emoticons   
from emoji import UNICODE_EMOJI         

@Language.factory("StandardTokenExtension")
def standard_spacy_token_ext(nlp, name, attrs=('is_emoticon','is_emoji')):
    """Get standard extensions for spacy Token object 
    
    Loaded in all use cases. Needed for multiple feature packages.

    Parameters
    ----------        
    nlp : spaCy Language object
    name: str
        name of spaCy extension (StandardTokenExtension)
    attrs : tuple, default=('is_emoticon','is_emoji')
        attributes to set on Token
                
    Returns
    -------
    StandardSpacyTokenExt object
    """
    return StandardSpacyTokenExt(nlp, attrs)    


class StandardSpacyTokenExt():
    """SpaCy Extension for basic Token attributes

    Parameters
    ----------        
    nlp : spaCy Language object
    attrs : tuple
        attributes to set on Token    
    
    Attributes
    ----------
    _pmatcher : spacy PhraseMatcher object
        matcher to match words from lexicon
    """
    def __init__(self, nlp, attrs):
        self._is_emoticon, self._is_emoji = attrs
        
        if not Token.has_extension(self._is_emoticon):
            Token.set_extension(self._is_emoticon, default=False)
        if not Token.has_extension(self._is_emoji):
            Token.set_extension(self._is_emoji, default=False)
        
        # Phrase Matcher (efficient with larger terminology lists, use matcher for small lists to match)
        self._pMatcher = PhraseMatcher(nlp.vocab)
        self._pMatcher.add(self._is_emoticon, list(nlp.tokenizer.pipe(emoticons)), on_match=self._set_feature_token)
        self._pMatcher.add(self._is_emoji, list(nlp.tokenizer.pipe(e.replace(' ','') for e in UNICODE_EMOJI.keys())), on_match=self._set_feature_token)
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
        _ = self._pMatcher(doc)
        return doc
    
    def _set_feature_token(self, matcher, doc, i, matches):
        """Set attribute specified for each match found

        Parameters
        ----------
        matcher : spaCy PhraseMatcher object
        doc : spaCy doc
        i : int
            index of match in matchess
        matches : list
            list of `(match_id, start, end)` tuples, describing the matches.                                     
        """
        matchId, start, end = matches[i]
        for token in doc[start:end]:
            token._.set(doc.vocab.strings[matchId], True)