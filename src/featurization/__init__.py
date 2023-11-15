import collections
from collections import Counter, defaultdict
import importlib
from utils.preprocess_data import get_texts
from .vectorizers import *
import pickle
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


class Featurization():
    """Featurization of corpora 

    Load featurization pipelines for different languages.
    Ensure constituency between different languages and constituency of configuration if pretrained models are used.
    Perform vectorization of raw output of feature packages.
    Load respective feature scalers.
    Perform basic visualization of features

    Parameters
    ----------
    config : dict

    Attributes
    ----------
    feature_packages : list
        list of dictionaries with feature packages configurations
    corpora : list
        list of dictionaries with corpora configurations
    corpora_files_ids : dict
        dictionary with dictionaries for each corpus containing file as key and index of file as value
    vectorized_features : dict
        dictionary containing feature package name or combination of feature package name and feature name as key
        and respective vectorized features as value
    feature_names : dict
        dictionary containing a list of the feature names for each feature package
    feature_names_list : list
        list of all feature names
    _base_dir : str
        path to base directory for output
    _lang_featurizers : dict
        dictionary with featurization pipeline for each language, if language not implemented yet generic featurizer (Featurizer super class in featurization.featurizers) is loaded
    _scorer_feature_packages : set
        feature packages that return list of scores (different visualization), extend if necessary
    """

    def __init__(self, config):
        self.feature_packages = config['feature_packages']
        self.corpora = config['corpora']
        self.corpora_files_ids = {}
        self._base_dir = config['base_dir']
        self._set_names_feature_packages()
        feature_packages_types = self._get_feature_packages_grouped_by_type()
        self._lang_featurizers = {}
        for lang in set([corp['language'] for corp in self.corpora]):
            try:
                self._lang_featurizers[lang] = getattr(importlib.import_module('featurization.featurizers.lang.' + lang), 'Featurizer')(feature_packages_types, **config.get(lang, {}))
            except ModuleNotFoundError:
                self._lang_featurizers[lang] = getattr(importlib.import_module('featurization.featurizers'),'Featurizer')(feature_packages_types, lang, **config.get(lang,{}))
        self._check_consistency_feature_packages()
        self._scorer_feature_packages = {'emotion','formality'} # feature packages that return list of scores (different visualization), extend here if necessary

    def get_features(self, dump_vectorizers=True, pretrained=False):
        """Get vectorized features

        Parameters
        ----------
        dump_vectorizers : bool, default=True
            if True vectorizers will be dumped
        pretrained : bool, default=False
            if True mode with pretrained model(s) is used and we have to load pretrained scalers, vectorizers and configuration.

        Returns
        -------
        dict
        dictionary containing feature package name or combination of feature package name and feature name as key
        """
        if pretrained:
            return self._vectorize_pretrained(self._analyze())
        else:
            return self._vectorize(self._analyze(), dump_vectorizers=dump_vectorizers)

    def get_scalers(self, path=None):
        """Get scalers

        Get scalers to scale features as defined for each feature package

        Parameters
        ----------
        path : None or str
            path to file with pretrained scalers if pretrained model is or pretrained models are used

        Returns
        -------
        dict
            dictionary with name of feature package or name of feature package in combination with feature name as keys and scalers as values
            ({feature_package_name:scalers})
        """
        if path:
            return self._load_pretrained_scalers(path)
        else:
            return self._get_scalers()

    def visualize_features(self, y=None, per='corpus'):
        """Visualize average feature values for specific feature packages

        Visualize average feature values per corpus or per target values (y)

        Parameters
        ----------
        y : None or array_like
        per : str, default='corpus
        """

        if not (y is None) and per != 'corpus':
            self._visualize_features_targets(y, per=per)
        else:
            self._visualize_features_corpus()

    def _analyze(self):
        """Analyze corpora

        Analyze corpora separately in respect to language.
        Save dictionary with dictionaries for each corpus containing file as key and index of file as value
        Assert corpus names are unique
        Assert corpus path contains files

        Returns
        -------
        dict
            dictionary with name of feature package as key
            and list of the feature package analysis per file as value
            
        Raises
        ------
        AssertionError
            if corpus names not unique
            if corpus path contains no files
        """
        analysis = {feature_package['name']: [] for feature_package in self.feature_packages}
        idx = 0
        for corpus in self.corpora:
            if not 'name' in corpus:
                corpus['name'] = os.path.basename(corpus['path']) 
            assert corpus['name'] not in self.corpora_files_ids, "Corpora names must be unique.  " \
                                                                 "If you did not name your corpus make sure the corpus' folder name is unique or set a corpus name. " \
                                                                 "Duplicate name: {} ".format(corpus['name'])
            print('Analyzing corpus {} ({}) ...'.format(corpus['name'], corpus['language']))
            files = get_texts(corpus['path'])
            assert files, "No files found in folder or file of corpus {}. Ensure you inserted the right path.".format(corpus['name'])
            self.corpora_files_ids[corpus['name']] = {file: i for file, i in zip(files.keys(), range(idx, idx + len(files)))}
            self._lang_featurizers[corpus['language']].featurize_files(analysis, files)
            idx += len(files)
        return analysis

    def _vectorize(self, analysis, dump_vectorizers=True):
        """Vectorize features of each feature package

        Get dictionary of vectorized features per feature package or feature of feature package (if feature package returns multiple features as list).
        Get dictionary with feature names per feature package or feature of feature package (if feature package returns multiple features as list).
        Get list of all feature names.

        Support list (use VectorizerFrequency_sklearn), scalar values (VectorizerDict_sklearn) and embedding stacking (EmbeddingsVectorizer) for feature return values in feature modules
        If your own feature module needs special vectorization extend if/else in self._vectorize and self._vectorize_pretrained
        and add vectorizer to vectorizers.py and self.vectorizers which is saved as pickle for further usage of models.

        Parameters
        ----------
        analysis : dict
            dictionary with name of feature package as key
            and list of the feature package analysis per file as value
        dump_vectorizers : bool
            if True vectorizers will be dumped

        Returns
        -------
        dict
            dictionary containing feature package name or combination of feature package name and feature name as key
        
        Warns
        -----
        UserWarning
            feature packages that returned empty values will be ignored
        """
        vectorizers = {} # dict with vectorizer per feature package or per feature of feature package (if multiple lists returned by feature package, key: featurePackageName_featureName)
        self.vectorized_features = {} # {feature_package:vectorized_features}
        self.feature_names = {} # {feature_package:list_of_feature_names}
        self.feature_names_list = [] # list of all feature names

        empty_feature_packages = []
        for feature_config in self.feature_packages:
            if feature_config['feature_package'] == 'embeddings': # Embeddings 
                embs = EmbeddingsVectorizer().vectorize([doc[feature_config['name']] for doc in analysis[feature_config['name']]])
                if embs.shape[1] != 0:
                    self.vectorized_features[feature_config['name']] = embs
                    self.feature_names[feature_config['name']] = ['emb_dim_{}'.format(i) for i in range(self.vectorized_features[feature_config['name']].shape[1])]
                    self.feature_names_list.extend(self.feature_names[feature_config['name']])
                else:
                    warnings.warn('Feature package {} returned empty values. It will be ignored.'.format(feature_config['name']))    
                    empty_feature_packages.append(feature_config['name'])                    
                
            elif 'vectorizer' in feature_config:  # VectorizerFrequency_sklearn for lists
                if 'features' in feature_config: # for feature packages which return lists for each feature
                    empty_features = []
                    for feature in feature_config['features']:
                        x = [doc[feature] for doc in analysis[feature_config['name']]]
                        if any(doc_features for doc_features in x): # check if feature does not include only empty lists
                            vectorizer = VectorizerFrequency_sklearn(x=x, **feature_config['vectorizer'])
                            feature_package_name = '{}_{}'.format(feature_config['name'], feature)
                            self.vectorized_features[feature_package_name] = vectorizer.vectorize(x)
                            vectorizers[feature_package_name] = vectorizer.vectorizer
                            self.feature_names[feature_package_name] = vectorizer.vectorizer.get_feature_names_out()
                            self.feature_names_list.extend(['{}_{}'.format(feature_package_name, feature_name) for feature_name in self.feature_names[feature_package_name]])
                            self._dump_features(feature_package_name, self.vectorized_features[feature_package_name], self.feature_names[feature_package_name])
                        else:
                            warnings.warn('Feature {} returned empty values. It will be ignored.'.format(feature))
                            empty_features.append(feature)
                    if empty_features:
                        feature_config['features'] = [feature for feature in feature_config['features'] if not feature in empty_features]
                     
                else: # for feature packages which return one list for feature package name
                    x = [doc[feature_config['name']] for doc in analysis[feature_config['name']]]
                    if any(doc_features for doc_features in x): # check if feature package does not include only empty lists
                        vectorizer = VectorizerFrequency_sklearn(x=x, **feature_config['vectorizer'])
                        self.vectorized_features[feature_config['name']] = vectorizer.vectorize(x)
                        vectorizers[feature_config['name']] = vectorizer.vectorizer
                        self.feature_names[feature_config['name']] = vectorizer.vectorizer.get_feature_names_out()
                        self.feature_names_list.extend(['{}_{}'.format(feature_config['name'], feature_name) for feature_name in self.feature_names[feature_config['name']]])
                        self._dump_features(feature_config['name'], self.vectorized_features[feature_config['name']], self.feature_names[feature_config['name']])
                    else:
                        warnings.warn('Feature package {} returned empty values. It will be ignored.'.format(feature_config['name']))    
                        empty_feature_packages.append(feature_config['name'])                 

            else:  # VectorizerDict_sklearn for scalar values
                if any(doc for doc in analysis[feature_config['name']]): # check if feature package does not include only empty dicts
                    vectorizer = VectorizerDict_sklearn(x=analysis[feature_config['name']])
                    self.vectorized_features[feature_config['name']] = vectorizer.vectorize(analysis[feature_config['name']])
                    vectorizers[feature_config['name']] = vectorizer.vectorizer
                    self.feature_names[feature_config['name']] = vectorizer.vectorizer.get_feature_names_out()
                    self.feature_names_list.extend(['{}_{}'.format(feature_config['name'], feature_name) for feature_name in self.feature_names[feature_config['name']]])
                    self._dump_features(feature_config['name'], self.vectorized_features[feature_config['name']], self.feature_names[feature_config['name']])
                else:
                    warnings.warn('Feature package {} returned empty values. It will be ignored.'.format(feature_config['name']))    
                    empty_feature_packages.append(feature_config['name'])  
            
        if empty_feature_packages:
            self.feature_packages = [feature_config for feature_config in self.feature_packages if not feature_config['name'] in empty_feature_packages]
        if dump_vectorizers:
            self._dump_vectorizers(vectorizers)
        return self.vectorized_features

    def _vectorize_pretrained(self, analysis):
        """Vectorize features of each feature package with pretrained vectorizers

        Parameters
        ----------
        analysis : dict
            dictionary with name of feature package as key
            and list of the feature package analysis per file as value

        Returns
        -------
        dict
            dictionary containing feature package name or combination of feature package name and feature name as key
        """
        vectorizers = self._load_pretrained_vectorizers()
        self._check_consistency_feature_packages_pretrained_config()
        self.vectorized_features = {}
        self.feature_names = {}
        self.feature_names_list = []

        for feature_config in self.feature_packages:
            if feature_config['feature_package'] == 'embeddings': # Embeddings
                self.vectorized_features[feature_config['name']] = EmbeddingsVectorizer().vectorize([doc[feature_config['name']] for doc in analysis[feature_config['name']]])
                self.feature_names[feature_config['name']] = ['emb_dim_{}'.format(i) for i in range(self.vectorized_features[feature_config['name']].shape[1])]
                self.feature_names_list.extend(self.feature_names[feature_config['name']])
                
            elif 'vectorizer' in feature_config: # VectorizerFrequency_sklearn for lists
                if 'features' in feature_config: # for feature packages which return lists for each feature
                    for feature in feature_config['features']:
                        feature_package_name = '{}_{}'.format(feature_config['name'], feature)
                        vectorizer = VectorizerFrequency_sklearn(vectorizer=vectorizers[feature_package_name])
                        self.vectorized_features[feature_package_name] = vectorizer.vectorize([doc[feature] for doc in analysis[feature_config['name']]])
                        self.feature_names[feature_package_name] = vectorizer.vectorizer.get_feature_names_out()
                        self.feature_names_list.extend(['{}_{}'.format(feature_package_name, feature_name) for feature_name in self.feature_names[feature_package_name]])
                        self._dump_features(feature_package_name, self.vectorized_features[feature_package_name], self.feature_names[feature_package_name])
                        
                else: # for feature packages which return one list for feature package name
                    vectorizer = VectorizerFrequency_sklearn(vectorizer=vectorizers[feature_config['name']])
                    self.vectorized_features[feature_config['name']] = vectorizer.vectorize([doc[feature] for doc in analysis[feature_config['name']]])
                    self.feature_names[feature_config['name']] = vectorizer.vectorizer.get_feature_names_out()
                    self.feature_names_list.extend(['{}_{}'.format(feature_config['name'], feature_name) for feature_name in self.feature_names[feature_config['name']]])
                    self._dump_features(feature_config['name'], self.vectorized_features[feature_config['name']], self.feature_names[feature_config['name']])
                    
            else: # VectorizerDict_sklearn
                vectorizer = VectorizerDict_sklearn(vectorizer=vectorizers[feature_config['name']])
                self.vectorized_features[feature_config['name']] = vectorizer.vectorize(analysis[feature_config['name']])
                self.feature_names[feature_config['name']] = vectorizer.vectorizer.get_feature_names_out()
                self.feature_names_list.extend(['{}_{}'.format(feature_config['name'], feature_name) for feature_name in self.feature_names[feature_config['name']]])
                self._dump_features(feature_config['name'], self.vectorized_features[feature_config['name']], self.feature_names[feature_config['name']])
        return self.vectorized_features

    def _get_scalers(self):
        """Initialize scaler for each feature package if defined

        Returns
        -------
        dict
            dictionary with feature package name or combination of feature package name and features as key
            and scaler object as value
        """
        scalers = {}
        for feature_config in self.feature_packages:
            if feature_config.get('scaler', False):
                if feature_config['name'] in self.vectorized_features:
                    scalers[feature_config['name']] = self._import_dynamically('sklearn.preprocessing', 
                                                                               feature_config['scaler']['name'])(**{k: v for k, v in feature_config['scaler'].items() if k != 'name'})
                else:
                    for feature in [feature for feature in self.vectorized_features
                                    if feature.startswith(feature_config['name'])]:
                        scalers['{}_{}'.format(feature_config['name'], feature)] = self._import_dynamically('sklearn.preprocessing', 
                                                                                                            feature_config['scaler']['name'])(**{k: v for k, v in feature_config['scaler'].items() if k != 'name'})
        return scalers

    def _import_dynamically(self, module, scaler):
        """Import scalers dynamically

        Parameters
        ----------
        module : str
            name of sklearn module to import class form
        model : str
            name of scaler class

        Returns
        -------
        class
            class in sklearn module to import

        Raises
        ------
        AttributeError
            if class not found
        """
        try:
            return getattr(importlib.import_module(module), scaler)
        except AttributeError as e:
            raise AttributeError("The chosen scaler {} does not exist in sklearn library. "
                                 "Make sure the name parameter doesn't have any typos.").format(scaler) from e

    def _dump_features(self, feature_package, features, feature_names):
        """Dump features as tsv or excel file
        
        file is only dumped for feature packages with less than 250 features

        Parameters
        ----------
        feature_package : str
            name of the feature package
        features : ndarray
            vectorized features
        feature_names : list
            list of the feature names
        """
        for corpus, file_ids in self.corpora_files_ids.items():
            feat_pack_features = features[[i for i in file_ids.values()], :]
            if feat_pack_features.shape[1]<250:
                features_corpus = pd.DataFrame(features[[i for i in file_ids.values()], :], index=file_ids.keys(),
                                               columns=[str(feature_name) for feature_name in feature_names])
                features_corpus.to_excel(os.path.join(self._base_dir, '{}_{}_analysis.xlsx'.format(corpus, feature_package)))
                features_corpus.to_csv(os.path.join(self._base_dir, '{}_{}_analysis.tsv'.format(corpus, feature_package)), sep='\t')

    def _visualize_features_corpus(self):
        """Visualize average feature values for specific feature packages corpus based
        """
        for feature_config in self.feature_packages:
            if feature_config['feature_package'] == 'embeddings':
                continue
            if not 'vectorizer' in feature_config:
                self._plot_labels(self._get_avg_feature_values_per_corpus(feature_config['name'], self.vectorized_features[feature_config['name']], self.feature_names[feature_config['name']]),
                                       os.path.join(self._base_dir, 'distro_{}_{}.png'.format(feature_config['name'], '_'.join(corpus['name'] for corpus in self.corpora))))
            elif feature_config['feature_package'] in self._scorer_feature_packages:
                for feature in [feature for feature in self.vectorized_features if feature.startswith(feature_config['name'])]:
                    self._plot_continuous(self._get_avg_feature_values_per_corpus(feature, self.vectorized_features[feature], self.feature_names[feature]),
                                          os.path.join(self._base_dir, 'distro_{}_{}.png'.format(feature, '_'.join(corpus['name'] for corpus in self.corpora))))

    def _visualize_features_targets(self, y, per='class'):
        """Visualize average feature values for specific feature packages based on targets (class or scores)

        Parameters
        ----------
        y : array_like
            target values
        per : {'class','score')
            if y is continuous or categorical
        """
        for feature_config in self.feature_packages:
            if feature_config['feature_package'] == 'embeddings':
                continue
            if not 'vectorizer' in feature_config:
                if per == 'score':
                    df = self._get_feature_values_per_score(feature_config['name'], self.vectorized_features[feature_config['name']], self.feature_names[feature_config['name']], y)
                    self._plot_scores(df, os.path.join(self._base_dir, 'distro_{}_per_{}_{}.png'.format(feature_config['name'], per, '_'.join(corpus['name'] for corpus in self.corpora))))
                else:
                    df = self._get_avg_feature_values_per_label(feature_config['name'], self.vectorized_features[feature_config['name']], self.feature_names[feature_config['name']], y)
                    self._plot_labels(df, os.path.join(self._base_dir, 'distro_{}_per_{}_{}.png'.format(feature_config['name'], per, '_'.join(corpus['name'] for corpus in self.corpora))))
            elif feature_config['feature_package'] in self._scorer_feature_packages:
                for feature in [feature for feature in self.vectorized_features if feature.startswith(feature_config['name'])]:
                    if per == 'score':
                        df = self._get_avg_feature_values_per_score_range(feature, self.vectorized_features[feature], self.feature_names[feature], y)
                    else:
                        df = self._get_avg_feature_values_per_label(feature, self.vectorized_features[feature], self.feature_names[feature], y)
                    self._plot_continuous(df, os.path.join(self._base_dir, 'distro_{}_per_{}_{}.png'.format(feature, per, '_'.join(corpus['name'] for corpus in self.corpora))))

    def _set_names_feature_packages(self):
        """Set names for feature packages

        Default is feature package name.
        Assert that no duplicate name is used

        Raises
        ------
        AssertionError
            if duplicate found
        """
        for feature_package in self.feature_packages:
            if not 'name' in feature_package:
                feature_package['name'] = feature_package['feature_package']
        duplicates = [k for k, v in Counter([feature_package['name'] for feature_package in self.feature_packages]).items() if v > 1]
        assert not duplicates, 'Names for feature packages must be unique. If you did not set a name of the feature package (its type) is used. Ensure to set an unique name for different feature packages of the same type. Duplicate names: {} '.format(', '.join(duplicates))

    def _get_feature_packages_grouped_by_type(self):
        """Get feature packages grouped by type and sorted by feature package type and name

        Returns
        -------
        dict
        """
        feature_packages_type = defaultdict(dict)
        for feature_package in sorted(self.feature_packages, key=lambda k: (k['feature_package'], k['name'])):
            feature_packages_type[feature_package['feature_package']][feature_package['name']] = {k:v for k,v in feature_package.items() if k != 'feature_package'}
        return feature_packages_type

    def _check_consistency_feature_packages(self):
        """Check consistency of feature packages regarding configuration and different language featurizers

        Ensure different languages contain same feature packages in the same order and
        ensure that these feature packages contain the same features in the features entry in the config
        (e.g. the emotion package doesn't have subjectivity implemented for German, so we can't consider it for English either)

        Raises
        ------
        Exception
            if configuration of feature packages are not consistent

        Warns
        --------
        UserWarning
            if a feature package is not implemented for one language, it will be ignored for all languages for further processing

        """
        sorted_featurizers = sorted(self._lang_featurizers, key=lambda key: len(self._lang_featurizers[key].feature_modules))
        # set reference config
        feature_names_not_implemented = set([feature_package['name'] for feature_package in self.feature_packages]) - set(self._lang_featurizers[sorted_featurizers[0]].feature_modules)

        if feature_names_not_implemented:
            warnings.warn('At least for language {} the following feature packages have not been implemented yet and '
                          'will be ignored: {}'.format(sorted_featurizers[0], ' / '.join(feature_names_not_implemented)))

        feature_packages = {feature_package['name']: feature_package for feature_package in self.feature_packages}

        for name, feature_module in self._lang_featurizers[sorted_featurizers[0]].feature_modules.items():
            feature_packages[name].update(feature_module.config)
        self.feature_packages = [feature_packages[feature_module] for feature_module in self._lang_featurizers[sorted_featurizers[0]].feature_modules]

        # check feature package order and consistency for different language featurizers
        for lang in sorted_featurizers[1:]:
            self._lang_featurizers[lang].feature_modules = {feature_package['name']: self._lang_featurizers[lang].feature_modules[feature_package['name']] for feature_package in self.feature_packages}
            # control configuration consistency and remove features not set in some language feature packages
            for config, ref_module, feature_module in zip(self.feature_packages,
                                                          self._lang_featurizers[sorted_featurizers[0]].feature_modules.values(),
                                                          self._lang_featurizers[lang].feature_modules.values()):
                if feature_module.config != ref_module.config:
                    try:
                        features = [set(lang_featurizer.feature_modules[config['name']].config['features'])
                                    for lang_featurizer in self._lang_featurizers.values()]
                        consistent_features = set.intersection(*features)
                        for lang_features, lang in zip(features, self._lang_featurizers):
                            if lang_features != consistent_features:
                                self._lang_featurizers[lang].feature_modules[config['name']].config['features'] = list(consistent_features)
                                print('Feature(s) {} of {} feature package in {} language package not consistent '
                                      'with other language feature packages: removed'.format(' / '.join(lang_features - consistent_features),
                                                                                             config['name'], lang))
                        config['features'] = list(consistent_features)
                    except:
                        raise Exception(
                            'Configuration of feature package {} in language package {} not consistent with other language packages. '
                            'Make sure the feature packages are comparable (e.g., named entity annotations comparable between languages). '
                            'If you implemented your own package make sure the configuration parameters are consistent through languages, otherwise contact support.'.format(config['name'], lang))

    def _check_consistency_feature_packages_pretrained_config(self):
        """Check consistency of feature package configuration with configuration of run of pretrained model(s)

        For now check if complete configuration matches (maybe edit for different path configurations etc. later on)

        Raises
        ------
        AssertionError
            if configuration for feature packages do not match
        """
        for feature_config, pretrained_config in zip(self.feature_packages, json.load(open(os.path.join(self._base_dir, 'config.json')))['feature_packages']):
            assert feature_config == pretrained_config, \
                'Feature packages of training and current configuration are not equal. Ensure the configurations match. {}'.format(feature_config['name']) \
                + json.dumps(pretrained_config, indent=4)

    def _load_pretrained_vectorizers(self):
        """Load pretrained vectorizers

        Parameters
        ----------
        input_path : str

        Returns
        -------
        dict
            dictionary with name of feature package or name of feature package in combination with feature name as keys and scalers as values
            ({feature_package_name:vectorizers})
        """
        return pickle.load(open(os.path.join(self._base_dir, 'vectorizers.pickle'), 'rb'))

    def _dump_vectorizers(self, vectorizers):
        """Dump vectorizers

        Parameters
        ----------
        vectorizers : dict
            dictionary with name of feature package or name of feature package in combination with feature name as keys and scalers as values
            ({feature_package_name:vectorizers})
        """
        pickle.dump(vectorizers, open(os.path.join(self._base_dir, 'vectorizers.pickle'), 'wb'))

    def _load_pretrained_scalers(self, input_path):
        """Load scalers of pretrained model

        Load scaler per feature package or features in feature packages, if scalers were trained

        Parameters
        ----------
        input_path : str
            path to load scalers from

        Returns
        -------
        dict
            dictionary with name of feature package or name of feature package in combination with feature name as keys and scalers as values
            ({feature_package_name:scalers})
        """
        input_path = input_path + '_scalers.pickle'
        return pickle.load(open(input_path, 'rb')) if os.path.exists(input_path) else {}

    # helpers visualization

    def _plot_continuous(self, df, file_name):
        """Plot continuous features per class or score range

        Parameters
        ----------
        df : pd DataFrame
        file_name : str
            path to output file
        """
        labels, values, corpus = df.columns
        plot = sns.lineplot(x=labels, y=values, hue=corpus, data=df, palette=sns.cubehelix_palette(n_colors=len(set([c for c in df[corpus]])), start=.5, rot=-.5))
        plt.savefig(file_name, dpi=300, format='png', bbox_inches='tight')
        plt.close()

    def _plot_scores(self, df, file_name):
        """Plot features per score

        Parameters
        ----------
        df : pd DataFrame
        file_name : str
            path to output file
        """
        
        labels, values, scores = df.columns
        plot = sns.lineplot(x=scores, y=values, hue=labels, data=df, palette=sns.cubehelix_palette(n_colors=len(set([c for c in df[labels]])), start=.5, rot=-.5))
        plt.savefig(file_name, dpi=300, format='png', bbox_inches='tight')
        plt.close()

    def _plot_labels(self, df, file_name):
        """Plot features per label/class

        Parameters
        ----------
        df : pd DataFrame
        file_name : str
            path to output file
        """
        labels, values, corpus = df.columns
        plot = sns.barplot(x=labels, y=values, hue=corpus, data=df, palette=sns.cubehelix_palette(n_colors=len(set([c for c in df[corpus]])), start=.5, rot=-.5))
        plot.set_xticklabels(plot.get_xticklabels(), rotation=70, ha="right", fontsize=7)
        plt.savefig(file_name, dpi=300, format='png', bbox_inches='tight')
        plt.close()

    def _get_avg_feature_values_per_corpus(self, feature_package, features, feature_names):
        """Calculate the average of feature values for each corpus

        Parameters
        ----------
        feature_package : str
            name of the feature package
        features : ndarray
            vectorized features
        feature_names : list
            list of the feature names

        Returns
        -------
        pd DataFrame
        """
        avg_feat_values = defaultdict(list)
        for corpus_name, file_indices in self.corpora_files_ids.items():
            avg_feat_values['Feature Package: {}'.format(feature_package)].extend(feature_names)
            avg_feat_values['Average Value'].extend(np.mean([features[i] for i in file_indices.values()], axis=0))
            avg_feat_values['Corpus'].extend([corpus_name] * len(feature_names))
        return pd.DataFrame(avg_feat_values)

    def _get_avg_feature_values_per_label(self, feature_package, features, feature_names, y):
        """Calculate the average of feature values for each label/class

        Parameters
        ----------
        feature_package : str
            name of the feature package
        features : ndarray
            vectorized features
        feature_names : list
            list of the feature names
        y : array_like
            target values

        Returns
        -------
        pd DataFrame
        """
        avg_feat_values = defaultdict(list)
        labels = np.unique(y).tolist()
        for label in labels:
            avg_feat_values['Feature Package: {}'.format(feature_package)].extend(feature_names)
            avg_feat_values['Average Value'].extend(np.mean([features[i] for i, label_y in enumerate(y) if label_y == label], axis=0))
            avg_feat_values['Label'].extend([label] * len(feature_names))
        return pd.DataFrame(avg_feat_values)

    def _get_avg_feature_values_per_score_range(self, feature_package, features, feature_names, y):
        """Calculate the average of feature values per score range

        For the score ranges the scores are mapped to 3 ranges

        Parameters
        ----------
        feature_package : str
            name of the feature package
        features : ndarray
            vectorized features
        feature_names : list
            list of the feature names
        y : array_like
            target values

        Returns
        -------
        pd DataFrame
        """
        avg_feat_values = defaultdict(list)
        start = y.min()
        end = y.max()

        diff = round((end - start) / 6, 2)
        lower_bound = round(start + diff, 2)
        upper_bound = round(end - diff, 2)

        for start, end, score_range_name in [
            (start, end, '{} – {}'.format(round(start - start / 100, 2), round(end + end / 100, 2))) for start, end in
            [(start, lower_bound), (lower_bound, upper_bound), (upper_bound, end)]]:
            avg_feat_values['Feature Package: {}'.format(feature_package)].extend(feature_names)
            avg_feat_values['Average Value'].extend(np.mean([features[i] for i, score in enumerate(y) if start <= score < end], axis=0))
            avg_feat_values['Score Range'].extend([score_range_name] * len(feature_names))
        return pd.DataFrame(avg_feat_values)

    def _get_feature_values_per_score(self, feature_package, features, feature_names, y):
        """Calculate the average of feature values per score 

        Parameters
        ----------
        feature_package : str
            name of the feature package
        features : ndarray
            vectorized features
        feature_names : list
            list of the feature names
        y : array_like
            target values

        Returns
        -------
        pd DataFrame
        """
        feat_values = defaultdict(list)
        for i, score in enumerate(y):
            feat_values['Feature Package: {}'.format(feature_package)].extend(feature_names)
            feat_values['Value'].extend(features[i])
            feat_values['Scores'].extend([score] * len(feature_names))
        return pd.DataFrame(feat_values)
