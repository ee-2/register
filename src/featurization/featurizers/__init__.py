from utils.preprocess_data import iter_texts
from .standard_spacy_doc_extension import standard_spacy_doc_ext
from .standard_spacy_span_extension import standard_spacy_span_ext
from .standard_spacy_token_extension import standard_spacy_token_ext
import spacy
from utils.custom_pipeline_components import *
import warnings

class Featurizer():
    """Language Independent Featurizer 
    
    Language independent featurizer for feature packages that require no special treatment per language.
    Super class for language dependent featurizers.
    
    Parameters
    ----------    
    feature_packages : dict 
        dictionary with feature packages and configuration
    lang : str
        ISO-code of language
    spacy_lm : str, default=''
        spacy language model (e.g. 'de_core_news_sm')
    custom_pipeline_components : dict, default={}
        dictionary with configuration for custom pipeline components to replace spaCy's components
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
    _ batch_size : int
        batch size for spaCy's nlp pipeline processing
    """
    def __init__(self, feature_packages, lang, spacy_lm='', custom_pipeline_components={}, n_process=1, batch_size=128):
        self._lang = lang
        self._set_nlp(spacy_lm, custom_pipeline_components=custom_pipeline_components, n_process=n_process, batch_size=batch_size)
        self._set_language_independent_feature_modules(feature_packages)

    def featurize_files(self, features, files):
        """Get features of files for each feature package
        
        Append respective features of feature module to list of the entry for each feature module in features

        Parameters
        ----------
        features : dict
            dictionary with list values
        files : dict
            dictionary with file names as keys and content as values
        """

        for doc in self.nlp.pipe(iter_texts(files.values()), n_process=self._n_process, batch_size=self._batch_size):
            for feature_module_name, feature_module in self.feature_modules.items():
                features[feature_module_name].append(feature_module.featurize_doc(doc))


    def _set_nlp(self, spacy_lm, custom_pipeline_components={}, n_process=1, batch_size=128):
        """Set spaCy nlp pipeline
        
        Set spaCy nlp configurations.
        Set custom pipeline components if defined in configuration.
        Add standard spaCy extensions which are used by multiple feature packages
        If spaCy does not provide a language model for a language a blank spaCy nlp pipeline is intialized (provides e.g. tokenization).
        
        Parameters
        ----------
        spacy_lm : str
            spacy language model (e.g. 'de_core_news_sm')
        custom_pipeline_components : dict, default={}
            dictionary with configuration for custom pipeline components to replace spaCy's components
        n_process : int, default=1
            number of processes for spaCy's nlp pipeline to use
        batch_size : int, default=128
            batch size for spaCy's nlp pipeline processing
        """
        self._n_process, self._batch_size = n_process, batch_size
        try:
            spacy_lm = spacy_lm if spacy_lm else self._lang + '_core_news_sm'
            self.nlp = spacy.load(spacy_lm, disable=[k for k in custom_pipeline_components.keys()])

        except:  # for languages with no models
            self.nlp = spacy.blank(self._lang)
            print('For Language {} no language model was found. Blank language model with limited support created.'.format(self._lang))

        if custom_pipeline_components: # set custom pipeline components
            if 'tokenizer' in custom_pipeline_components:
                self.nlp.tokenizer = custom_tokenizer(self.nlp)

            for pipeline_component in [comp for comp in ['tok2vec',
                                                         'tagger',
                                                         'morphologizer',
                                                         'parser',
                                                         'senter',
                                                         'ner',
                                                         'attribute_ruler',
                                                         'lemmatizer',
                                                         'textcat'] if comp in custom_pipeline_components]:
                self.nlp.add_pipe('custom_{}'.format(pipeline_component), config=custom_pipeline_components[pipeline_component])
        self.nlp.add_pipe('StandardDocExtension')
        self.nlp.add_pipe('StandardSpanExtension')
        self.nlp.add_pipe('StandardTokenExtension')


    def _set_language_independent_feature_modules(self, feature_packages):
        """Set language independent feature modules
        
        Basic language independent feature modules are added to self.feature_modules.
        Overwrite this method if your language has a special treatment for an otherwise language independent feature package
        
        Parameters
        ----------    
        feature_packages : dict 
            dictionary with feature packages and configuration        
        """
        self.feature_modules = {}

        if 'character_ngrams' in feature_packages:
            from .ngrams.char_ngrams import CharNgramsFeaturizer
            for feature_package, feature_package_config in feature_packages['character_ngrams'].items():
                self.feature_modules[feature_package] = CharNgramsFeaturizer(**feature_package_config)

        if 'token_ngrams' in feature_packages:
            from .ngrams.token_ngrams import TokenNgramsFeaturizer
            for feature_package, feature_package_config in feature_packages['token_ngrams'].items():
                self.feature_modules[feature_package] = TokenNgramsFeaturizer(**feature_package_config)

        if 'span_ngrams' in feature_packages:
            from .ngrams.span_ngrams import SpanNgramsFeaturizer
            for feature_package, feature_package_config in feature_packages['span_ngrams'].items():
                self.feature_modules[feature_package] = SpanNgramsFeaturizer(**feature_package_config)

        if 'named_entities' in feature_packages:
            from .namedentities import NamedEntityFeaturizer
            for feature_package, feature_package_config in feature_packages['named_entities'].items():
                self.feature_modules[feature_package] = NamedEntityFeaturizer(self.nlp, **feature_package_config)
            
        if 'embeddings' in feature_packages:            
            from .embeddings import EmbeddingsFeaturizer
            for feature_package, feature_package_config in feature_packages['embeddings'].items():
                print(feature_package_config)
                self.feature_modules[feature_package] = EmbeddingsFeaturizer(self._lang, **feature_package_config)

        if 'morphology' in feature_packages:
            from .morphology import MorphologyFeaturizer
            for feature_package, feature_package_config in feature_packages['morphology'].items():
                self.feature_modules[feature_package] = MorphologyFeaturizer(**feature_package_config)

        if 'orthography' in feature_packages:
            from .orthography import OrthographyFeaturizer
            for feature_package, feature_package_config in feature_packages['orthography'].items():
                self.feature_modules[feature_package] = OrthographyFeaturizer(**feature_package_config)

        if 'emotion' in feature_packages:
            from .emotion import EmotionFeaturizer
            for feature_package, feature_package_config in feature_packages['emotion'].items():
                self.feature_modules[feature_package] = EmotionFeaturizer(self.nlp, self._lang, **feature_package_config)
        
        if 'syntax' in feature_packages:
            from .syntax import SyntaxFeaturizer
            for feature_package, feature_package_config in feature_packages['syntax'].items():
                if feature_package_config.get('grammar','dependency') == 'constituency':
                    try:
                        if not 'benepar' in self.nlp.pipe_names:
                            import benepar
                            self.nlp.add_pipe('benepar', config={'model': f'benepar_{self._lang}3' if self._lang == 'en' else f'benepar_{self._lang}2'})
                    except LookupError as e:
                        warnings.warn(f'{e} '
                                      f'Feature package {feature_package["name"]} will be ignored.')
                        continue                        
                self.feature_modules[feature_package] = SyntaxFeaturizer(**feature_package_config)  
            
        if 'metrics' in feature_packages:
            from .metrics import MetricsFeaturizer
            from spacy_syllables import SpacySyllables
            self.nlp.add_pipe("syllables", after="tagger")
            for feature_package, feature_package_config in feature_packages['metrics'].items():
                if 'avg_constituency_height' in feature_package_config.get('features', []):
                    try:
                        if not 'benepar' in self.nlp.pipe_names:
                            import benepar
                            self.nlp.add_pipe('benepar', config={'model': f'benepar_{self._lang}3' if self._lang == 'en' else f'benepar_{self._lang}2'})
                    except LookupError as e:
                        warnings.warn(f'{e} '
                                      f'Feature avg_constituency_height in feature package {feature_package["name"]} will be ignored.')
                            
                        feature_package_config['features'] = [f for f in feature_package_config['features'] if f != 'avg_constituency_height']
                self.feature_modules[feature_package] = MetricsFeaturizer(self.nlp, **feature_package_config)

        """
        add language independent feature modules here 
        """