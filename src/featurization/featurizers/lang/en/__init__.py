from ... import Featurizer

class Featurizer(Featurizer):
    """English Featurizer
    
    Featurizer for feature packages with special treatment for English 
    Inherit from language independent featurizer.
    
    Parameters
    ----------    
    feature_packages : dict 
        dictionary with feature packages and configuration
    spacy_lm : str, default='en_core_web_sm'
        spacy language model to use
    custom_pipeline_components : dict, default={}
        dictionary with configuration for custom pipeline components
    n_process : int, default=1
        number of processes for spaCy's nlp pipeline to use
    batch_size : int, default=128
        batch size for spaCy's nlp pipeline processing
    
    Attributes
    ----------
    nlp : spaCy Language object
    feature_modules : dict
        dictionary with feature package name as key and feature package object as value        
    _lang : str
        ISO-code of language
    _n_process : int
        number of processes for spaCy's nlp pipeline to use
    _batch_size : int
        batch size for spaCy's nlp pipeline processing
    """    
    def __init__(self, feature_packages, spacy_lm='en_core_web_sm', custom_pipeline_components={}, n_process=1, batch_size=128):
        self._lang = 'en'
        self._set_nlp(spacy_lm, custom_pipeline_components=custom_pipeline_components, n_process=n_process, batch_size=batch_size)
        # remove feature packages to set them language dependently
        self._set_language_independent_feature_modules({k: v for k, v in feature_packages.items() if k not in ['emotion']})
        self._set_language_dependent_feature_modules(feature_packages)

    def _set_language_dependent_feature_modules(self, feature_packages):
        """Set language dependent feature modules
        
        Set emotion, readability and semantic_relations feature modules for English

        Parameters
        ----------    
        feature_packages : dict 
            dictionary with feature packages and configuration        
        """
        if 'emotion' in feature_packages:
            from .emotion import EmotionFeaturizer
            for feature_package, feature_package_config in feature_packages['emotion'].items():
                self.feature_modules[feature_package] = EmotionFeaturizer(self.nlp, **feature_package_config)

        """
        add your language dependent modules here 
        """