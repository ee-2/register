from ... import Featurizer

class Featurizer(Featurizer):
    """German Featurizer
    
    Featurizer for feature packages with special treatment for German 
    Inherit from language independent featurizer.
    
    Parameters
    ----------    
    feature_packages : dict 
        dictionary with feature packages and configuration
    spacy_lm : str, default='de_core_news_sm'
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
    def __init__(self, feature_packages, spacy_lm='de_core_news_sm', custom_pipeline_components={}, n_process=1, batch_size=128):
        self._lang = 'de'
        self._set_nlp(spacy_lm, custom_pipeline_components=custom_pipeline_components, n_process=n_process, batch_size=batch_size)
        # remove feature packages to set them language dependently
        self._set_language_independent_feature_modules({k: v for k, v in feature_packages.items() if k not in ['emotion','metrics']})
        self._set_language_dependent_feature_modules(feature_packages)      

    def _set_language_dependent_feature_modules(self, feature_packages):
        """Set language dependent feature modules
        
        Set emotion, formality, readability and semantic_relations feature modules for German

        Parameters
        ----------    
        feature_packages : dict 
            dictionary with feature packages and configuration        
        """
        if 'formality' in feature_packages:
            from .formality import FormalityFeaturizer
            for feature_package, feature_package_config in feature_packages['formality'].items(): 
                self.feature_modules[feature_package] = FormalityFeaturizer(self.nlp, **feature_package_config)        

        if 'emotion' in feature_packages:
            from .emotion import EmotionFeaturizer
            for feature_package, feature_package_config in feature_packages['emotion'].items():
                self.feature_modules[feature_package] = EmotionFeaturizer(self.nlp, **feature_package_config)

        if 'metrics' in feature_packages:
            from .metrics import MetricsFeaturizer
            from spacy_syllables import SpacySyllables
            self.nlp.add_pipe("syllables", after="tagger")
            for feature_package, feature_package_config in feature_packages['metrics'].items():
                if 'avg_constituency_height' in feature_package_config.get('features', []):
                    if not 'benepar' in self.nlp.pipe_names:
                        import benepar
                        self.nlp.add_pipe('benepar', config={'model': 'benepar_de2'})
                self.feature_modules[feature_package] = MetricsFeaturizer(self.nlp, **feature_package_config)     

        """
        define your language dependent modules here   
        """
