import os
import json
import pickle
from collections import defaultdict
from sklearn.model_selection import KFold, train_test_split
from featurization import Featurization
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from statistics import mean, stdev
import importlib
from scipy.stats import wilcoxon, ttest_rel
from itertools import combinations


class Register():
    """Main program
    
    Delegate analysis of corpora and computation of results for each mode,
    prepares preconditions (folds, y) for machine learning
    
    Parameters
    ----------
    config : dict
    
    Attributes
    ----------
    config : dict
        the configurations for the run
    featurization : Featurization
        object which performs the featurization of documents
    """

    def __init__(self, config):
        """Initialize featurization
        
        Parameters
        ----------
        config : dict
        """
        self.config = config
        self.featurization = Featurization(self.config)

    def cluster(self):
        """Cluster texts 
        
        Organize run per model: featurize, scale, encode, fit and predict, evaluate (optional), visualize results
        
        Returns
        -------
        pd DataFrame
            pd DataFrame with cluster for each file determined by respective clustering model
        """
        print('Performing clustering...')
        x = self.featurization.get_features()
        y = self._get_y_clustering()
        y = LabelEncoder().fit_transform([v['class'] for v in y.values()]) if y else None
        results_files = pd.DataFrame([{'doc_name': file, 'corpus_name': corpus} for corpus, files in self.featurization.corpora_files_ids.items() for file in files])
        results_files.set_index('doc_name', inplace=True)

        for clustering_model in self.config.get('models', [{"model": "AgglomerativeClustering", "lib": "sklearn"}]):
            print('Processing model {}'.format(clustering_model['model']))
            model = self._import_dynamically('models.clustering.clustering_{}'.format(clustering_model['lib']),
                                             'Clustering_{}'.format(clustering_model['lib']))(
                                                 scalers=self.featurization.get_scalers(), 
                                                 **clustering_model)
            x_scaled = model.scale(x)
            results_files[clustering_model['model']] = model.fit_predict(x_scaled)

            if not (y is None):
                results_model = {clustering_model['model']: model.eval(y, results_files[clustering_model['model']])}
                self.dump_results(pd.DataFrame(results_model), file_name='results_{}'.format(clustering_model['model']))
            if self.config.get("explanation_shap", False):
                model.explain(x_scaled, self.featurization.feature_names_list, clustering_model['model'], self.config['base_dir'])
            model.visualize(x_scaled, 
                            results_files[['corpus_name', clustering_model['model']]], 
                            self.featurization.corpora_files_ids, 
                            clustering_model['model'], 
                            self.config['base_dir'])
            self.featurization.visualize_features(y=results_files[clustering_model['model']], per='class')
        print('Done.')
        return results_files

    def classify(self):
        """Classify texts with pretrained models 
        
        Organize run per model: featurize, scale, encode, predict, visualize results
        
        Returns
        -------
        pd DataFrame
            pd DataFrame with class for each file determined by respective classification model
        """
        print('Classify texts...')
        x = self.featurization.get_features(pretrained=True)
        encoder = self._load_encoder()
        results = pd.DataFrame([{'doc_name':file, 'corpus_name':corpus} for corpus, files in self.featurization.corpora_files_ids.items() for file in files])
        results.set_index('doc_name',inplace=True)
        for classifier in self.config['models']:
            model = self._import_dynamically('models.classification.classification_{}'.format(classifier['lib']),
                                             'Classification_{}'.format(classifier['lib']))(
                                                 scalers=self.featurization.get_scalers(path=os.path.join(self.config['base_dir'], classifier['pretrained'])),                    
                                                 dim=len(self.featurization.feature_names_list), 
                                                 path=os.path.join(self.config['base_dir'], classifier['pretrained']),
                                                 nr_labels=encoder.classes_.size,
                                                 **classifier)   
            y_pred = model.predict(model.scale_apply(x))
            results[classifier['model']] = encoder.inverse_transform(y_pred.ravel())
            model.visualize_pretrained(results[['corpus_name',classifier['model']]], self.config['base_dir'], classifier['model'])
            self.featurization.visualize_features(y=results[classifier['model']], per='class')
        print('Done.')
        return results

    def score(self):
        """Score texts with pretrained models 
        
        Organize run per model: featurize, scale, predict, visualize results
        
        Returns
        -------
        pd DataFrame
            pd DataFrame with score for each file determined by respective linear regression model
        """
        print('Score texts...')
        x = self.featurization.get_features(pretrained=True)
        results = pd.DataFrame([{'doc_name':file, 'corpus_name':corpus} for corpus, files in self.featurization.corpora_files_ids.items() for file in files])
        results.set_index('doc_name',inplace=True)
        for lin_model in self.config['models']:
            model = self._import_dynamically('models.linearregression.linear_regression_{}'.format(lin_model['lib']),
                                             'LinearRegression_{}'.format(lin_model['lib']))(
                                                 scalers=self.featurization.get_scalers(path=os.path.join(self.config['base_dir'], lin_model['pretrained'])),
                                                 dim=len(self.featurization.feature_names_list), 
                                                 path=os.path.join(self.config['base_dir'], lin_model['pretrained']),
                                                 **lin_model)   
            y_pred = model.predict(model.scale_apply(x)).ravel()
            results[lin_model['model']]=y_pred
            model.visualize_pretrained(results[['corpus_name',lin_model['model']]], self.config['base_dir'], lin_model['model'])
            self.featurization.visualize_features(y=results[lin_model['model']], per='score')
        print('Done.')
        return results  

    def train_classifier(self):
        """Train classification models

        Organize run per model and fold: featurize, encode, scale, fit, predict, evaluate (if not only training data),
        visualize results (if not only training data)
        
        Returns
        -------
        dict or None
            dictionary with dictionaries for each model containing a dictionary of results per fold for each metric
            None if only training set is specified and model is trained solely without any test data
        """
        x = self.featurization.get_features()
        y = np.array([v['class'] for v in self._get_y().values()])
        folds = self._get_folds()
        self.featurization.visualize_features(y=y, per='class')

        # encoding y
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        target_names = list(encoder.classes_)
        labels = encoder.transform(target_names)
        self._dump_encoder(encoder)

        print('Training classifiers...')
        results = {}
        for classifier in self.config.get('models',[{"model":"SGDClassifier","lib":"sklearn"}]):
            print('Processing model {}'.format(classifier['model']))
            
            out_dir_model = os.path.join(self.config['base_dir'], classifier['model'])
            os.makedirs(out_dir_model, exist_ok=True)

            results_model = defaultdict(dict)
            y_pred_total = []
            y_test_total = []
            for fold, splits in folds.items():
                print('Processing {}'.format(fold))
                model = self._import_dynamically('models.classification.classification_{}'.format(classifier['lib']), 
                                                 'Classification_{}'.format(classifier['lib']))(
                                                     scalers=self.featurization.get_scalers(),                    
                                                     dim=len(self.featurization.feature_names_list),
                                                     nr_labels=len(target_names), 
                                                     **classifier)   
                x_train, y_train = model.scale_split_train(x, y, splits['train'])
                model.fit(x_train, y_train) 
                     
                if 'test' in splits:
                    x_test, y_test = model.scale_split_test(x, y, splits['test'])
                    y_pred = model.predict(x_test)
                    y_pred_total.extend(y_pred)
                    y_test_total.extend(y_test)
                    for metric, res in model.eval(y_test, y_pred, labels, target_names).items():
                        results_model[metric][fold] = res  
                    if self.config.get("explanation_shap",False):
                        model.explain(x_train, x_test, self.featurization.feature_names_list, '{}_{}'.format(classifier['model'], fold), out_dir_model)
                          
                model.dump('{}_{}'.format(classifier['model'], fold), out_dir_model)

            if results_model:
                if len(folds) > 1:
                    for metric, res in model.eval(y_test_total, y_pred_total, labels, target_names).items():
                        results_model[metric]['total'] = res
                else:
                    for metric in results_model:
                        results_model[metric]['total'] = results_model[metric][next(iter(folds))]
                model.visualize(results_model['classification_report']['total'], 
                                results_model['confusion_matrix']['total'], 
                                target_names, self.config['base_dir'], 
                                classifier['model'])
                results[classifier['model']] = results_model
                self.dump_results(pd.DataFrame(data=results_model), file_name='results_{}'.format(classifier['model']))
                print(f'Done: F Score {results[classifier["model"]]["f_score"]["total"]}')
        return results

    def train_linear_regressor(self):
        """Train linear regression models
         
        Organize run per model and fold: featurize, encode, scale, fit, predict, evaluate (if not only training data), visualize results (if not only training data)
        
        Returns
        -------
        dict or None
            dictionary with dictionaries for each model containing a dictionary of results per fold for each metric
            None if only training set is specified and model is trained solely without any test data
        """
        x = self.featurization.get_features()
        y = np.array([v['score'] for v in self._get_y().values()])
        folds = self._get_folds()
        self.featurization.visualize_features(y=y, per='score')
        print('Training linear regression models...')
        results = {}
        for lin_model in self.config.get('models', [{"model": "Ridge", "lib": "sklearn"}]):
            print('Processing model {}'.format(lin_model['model']))

            out_dir_model = os.path.join(self.config['base_dir'], lin_model['model'])
            os.makedirs(out_dir_model, exist_ok=True)

            results_model = defaultdict(dict)

            y_pred_total = []
            y_test_total = []
            for fold, splits in folds.items():
                print('Processing {}'.format(fold))
                model = self._import_dynamically('models.linearregression.linear_regression_{}'.format(lin_model['lib']), 
                                                 'LinearRegression_{}'.format(lin_model['lib']))(
                                                     scalers=self.featurization.get_scalers(),
                                                     dim=len(self.featurization.feature_names_list), 
                                                     **lin_model)   
                x_train, y_train = model.scale_split_train(x, y, splits['train'])
                model.fit(x_train, y_train)    
                        
                if 'test' in splits:
                    x_test, y_test = model.scale_split_test(x, y, splits['test'])
                    y_pred = model.predict(x_test)
                    y_pred_total.extend(y_pred)
                    y_test_total.extend(y_test)
                    for metric, res in model.eval(y_test, y_pred).items():
                        results_model[metric][fold] = res  
                    if self.config.get("explanation_shap",False):
                        model.explain(x_train, x_test, self.featurization.feature_names_list, '{}_{}'.format(lin_model['model'], fold), out_dir_model)
                    
                model.dump('{}_{}'.format(lin_model['model'], fold), out_dir_model)
                    
            if results_model:
                for metric,res in results_model.items():
                    avg = mean([r for r in res.values()])
                    standard_dev = 0.0 if len(folds) == 1 else stdev([r for r in res.values()])
                    results_model[metric]['avg'] = avg
                    results_model[metric]['stdev'] = standard_dev   
                results[lin_model['model']] = results_model
                self.dump_results(pd.DataFrame(data=results_model), file_name='results_{}'.format(lin_model['model']))
                model.visualize(y_test_total, y_pred_total, self.config['base_dir'], lin_model['model'])
                print(f'Done: Spearman Correlation {results[lin_model["model"]]["spearman_r"]["avg"]}')
        return results

    def analyze(self):
        """Delegate visualization of not further processed features 
        
        No machine learning is performed at all
        """
        print('Visualize features...')
        x = self.featurization.get_features(dump_vectorizers=False)
        self.featurization.visualize_features()

    def get_significance(self, results, metric='spearman_r', best_det='avg'):
        """Calculate significance 
        
        Calculate Wilcoxon signed-rank test and T-Test for training a classifier or linear regressor
        calculate significance pairwise with best model compared to all others
        
        Parameters
        ----------
        results : dict
            dictionary with dictionaries for each model containing a dictionary of results per fold for each metric
        metric : str, default='spearman_r'
            metric to calculate significance on
        best_det : str, default='avg'
            value on which to determine the best model (i.e. 'avg' for average of all runs or 'total' for total results)
            
        Returns
        -------
        pd DataFrame
            pd DataFrame with significance results      
        """
        results_significance = []
        models_sorted_best = {k:[res for fold, res in v[metric].items() if fold not in ['avg','stdev','total']]
                              for k,v in sorted(results.items(), key=lambda x:x[1][metric][best_det], reverse=True)}
        for combination in combinations(models_sorted_best.keys() , 2):
            first_model_scores = models_sorted_best[combination[0]]
            second_model_scores = models_sorted_best[combination[1]]
    
            wilc_stat, wilc_p = wilcoxon(first_model_scores, second_model_scores)
            t_stat, t_p = ttest_rel(first_model_scores, second_model_scores)
            results_significance.append({'First Model Name': combination[0], 
                                         'First Model {} {}'.format(best_det,metric):results[combination[0]][metric][best_det],
                                         'Second Model Name':combination[1],
                                         'Second Model {} {}'.format(best_det,metric):results[combination[1]][metric][best_det],
                                         'Wilcoxon Statistics':wilc_stat, 'Wilcoxon P': wilc_p,
                                         'T Statistics' : t_stat, 'T p':t_p})  
        return pd.DataFrame(results_significance)

    def dump_config(self):
        """Dump config file with updated feature packages
        
        Reset feature_packages in config to the ones determined in Featurization for constituency
        Reset corpora to potentially updated configuration (unique names) from Featurization
        """
        self.config['feature_packages'] = self.featurization.feature_packages
        self.config['corpora'] = self.featurization.corpora
        json.dump(self.config,
                  open(os.path.join(self.config['base_dir'], 'config.json'), 'w'),
                  ensure_ascii=False,
                  indent=4,
                  sort_keys=True)

    def dump_results(self, results, file_name='results'):
        """Dump results of models and significance tests 
        
        Parameters
        ----------
        results : pd DataFrame
        file_name : str, default='results'
            name of file to dump
        """
        results.to_csv(os.path.join(self.config['base_dir'], '{}.tsv'.format(file_name)), sep='\t', encoding='utf-8')
        results.to_excel(os.path.join(self.config['base_dir'], '{}.xlsx'.format(file_name)))

    def _get_y(self):
        """Get y
        
        Get target values to train models on: 
        target values are specified in a tsv file with id (name of document e.g.) in the first row and class/score,
        for classification the class can also be specified per corpus in the config file

        Returns
        -------
        dict
            dictionary with tuple of file and corpus name as key and target values as values
            
        Raises
        ------
        AssertionError
            if not all files have target values
        """
        y = {}
        for corpus in self.featurization.corpora:
            if 'class' in corpus:
                y.update({(file, corpus['name']): {'class': corpus['class']} for file in self.featurization.corpora_files_ids[corpus['name']].keys()})
            elif 'path_targets' in corpus:
                target_values = pd.read_csv(corpus['path_targets'], encoding='utf-8', index_col=0, sep='\t').to_dict(orient='index')
                y.update({(file, corpus['name']): target_values[file] for file in self.featurization.corpora_files_ids[corpus['name']].keys()})

        assert len(y) == len([file for corpus in self.featurization.corpora_files_ids.values() for file in corpus]), \
            'Make sure to provide target values for all files in your corpora if you want to train a model either via ' \
            'a tsv file under path_targets or by setting class in the corpus configuration.'
        return y

    def _get_y_clustering(self):
        """Get y for cluster mode
        
        Get class labels to evaluate clustering model on
        classes are specified in a tsv file with id (name of document e.g.) in the first row and class or
        specified per corpus in the config file

        Returns
        -------
        dict
            dictionary with tuple of file and corpus name as key and target values as values,
            if class isn't found for all files return empty dictionary
        """
        y = {}
        for corpus in self.featurization.corpora:
            if 'class' in corpus:
                y.update({(file, corpus['name']): {'class': corpus['class']} for file in self.featurization.corpora_files_ids[corpus['name']].keys()})
            elif 'path_targets' in corpus:
                target_values = pd.read_csv(corpus['path_targets'], encoding='utf-8', index_col=0, sep='\t').to_dict(orient='index')
                y.update({(file, corpus['name']): target_values[file] for file in self.featurization.corpora_files_ids[corpus['name']].keys()})

        if len(y) != len([file for corpus in self.featurization.corpora_files_ids.values() for file in corpus]) or any('class' not in v for v in y.values()):
            y = {}
        return y

    def _get_folds(self):
        """Get folds/splits
        
        Get train and test folds/splits:
        specified per corpus in config file under set and/or
        set per file in a tsv file containing columns starting with fold and train or test for each file
        (only for train and test) or set randomly
        
        Returns
        -------
        dict
            dictionary with dictionaries of sets with file ids as values for each fold
        """
        x_id = {(file, corpus): idx for corpus, files in self.featurization.corpora_files_ids.items() for file, idx in
                files.items()}
        folds = defaultdict(lambda: defaultdict(list))
        fixed_sets = defaultdict(list)
        fold_files = set()
        for corpus in self.featurization.corpora:
            if 'set' in corpus and corpus['set'] in ['train', 'test']:
                files = [(file, corpus['name']) for file in self.featurization.corpora_files_ids[corpus['name']].keys()]
                fixed_sets[corpus['set']].extend(
                    [idx for idx in self.featurization.corpora_files_ids[corpus['name']].values()])
                fold_files.update(files)
            elif corpus.get('path_train_test_splits'):
                df = pd.read_csv(corpus['path_train_test_splits'], encoding='utf-8', index_col=0, sep='\t')
                for column in df:
                    if column.startswith('fold'):
                        train = [(file, corpus['name']) for file in df.index[df[column] == 'train']]
                        test = [(file, corpus['name']) for file in df.index[df[column] == 'test']]
                        folds[column]['train'].extend([x_id[file] for file in train])
                        folds[column]['test'].extend([x_id[file] for file in test])
                        fold_files.update(train)
                        fold_files.update(test)

        # combine set definitions and definitions from tsv files
        if fixed_sets and folds:
            for dataset, indices in fixed_sets.items():
                for splits in folds.values():
                    splits[dataset].extend(indices)

        # all splits defined via set or tsv file
        if len(fold_files) == len(x_id):
            # all splits defined via set
            if not folds:
                return {'fold_0': fixed_sets}
                # splits either defined via tsv or via tsv and set
            else:
                return folds

        # some files predefined via tsv or set
        if folds:
            non_fold_files = [k for k in x_id.keys() if k not in fold_files]
            if len(folds) == 1:
                train, test = train_test_split(non_fold_files, shuffle=True, random_state=42)
                for splits in folds.values():
                    splits['train'].extend([x_id[file] for file in train])
                    splits['test'].extend([x_id[file] for file in test])
            else:
                for fold, (train, test) in zip(folds.keys(), KFold(n_splits=len(folds), shuffle=True, random_state=42).split(non_fold_files)):
                    folds[fold]['train'].extend([x_id[non_fold_files[i]] for i in train])
                    folds[fold]['test'].extend([x_id[non_fold_files[i]] for i in test])
                    # some files predefined via set
        elif fixed_sets:
            non_fold_files = [k for k in x_id.keys() if k not in fold_files]
            n_splits = self.config.get('cv_n_splits', 10)
            if n_splits <= 1:
                train, test = train_test_split(non_fold_files, shuffle=True, random_state=42)
                folds['fold_0']['train'] = [x_id[file] for file in train]
                folds['fold_0']['test'] = [x_id[file] for file in test]
                for dataset, indices in fixed_sets.items():
                    folds['fold_0'][dataset].extend(indices)
            else:
                for fold, (train, test) in enumerate(
                        KFold(n_splits=n_splits, shuffle=True, random_state=42).split(non_fold_files)):
                    folds['fold_{}'.format(fold)]['train'] = [x_id[non_fold_files[i]] for i in train]
                    folds['fold_{}'.format(fold)]['test'] = [x_id[non_fold_files[i]] for i in test]
                for dataset, indices in fixed_sets.items():
                    for splits in folds.values():
                        splits[dataset].extend(indices)
                        # no files predefined for splitting
        else:
            n_splits = self.config.get('cv_n_splits', 10)
            if n_splits <= 1:
                train, test = train_test_split([k for k in x_id.keys()], shuffle=True, random_state=42)
                folds = {'fold_0': {'train': [x_id[file] for file in train], 'test': [x_id[file] for file in test]}}
            else:
                folds = {'fold_{}'.format(fold): {'train': train, 'test': test} for fold, (train, test) in
                         enumerate(KFold(n_splits=n_splits, shuffle=True, random_state=42).split([k for k in x_id.keys()]))}
        return folds

    def _import_dynamically(self, module, model):
        """Import models dynamically 
        
        Parameters
        ----------        
        module : str
            name of module to import class form
        model : str
            name of model class
            
        Returns
        -------
        class
            class in module to import    
               
        Raises
        ------
        AttributeError
            if class not found
        """
        try:
            return getattr(importlib.import_module(module), model)
        except AttributeError as e:
            raise AttributeError(
                "Register does not provide model {}. See the docs for implementing your own model class").format(
                model) from e

    def _dump_encoder(self, encoder):
        """Dump encoder

        Parameters
        ----------        
        encoder : encoder object
            sklearn.preprocessing.LabelEncoder
        """
        pickle.dump(encoder, open(os.path.join(self.config['base_dir'], 'encoder.pickle'), 'wb'))

    def _load_encoder(self):
        """Load pretrained encoder of y 

        Returns
        -------
        encoder object
            sklearn.preprocessing.LabelEncoder     
        """
        return pickle.load(open(os.path.join(self.config['base_dir'], 'encoder.pickle'), 'rb'))
