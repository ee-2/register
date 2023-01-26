from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import pickle, os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Classification():
    """Classification super class
    
    Only used for inheritance of functions to different classification models (sklearn or tf_keras)

    See Also
    --------
    models.classification.classification_tf_keras
    models.classification.classification_sklearn
    """    
    
    def fit(self, x_train, y_train):
        """Fit model 
        
        Parameters
        ----------
        x_train : ndarray
        y_train : ndarray
        """
        self.model.fit(x_train, y_train)
    
    def predict(self, x_test):
        """Predict classes 
        
        Parameters
        ----------
        x_test : ndarray
                    
        Returns
        -------
        array_like
        """
        return self.model.predict(x_test)
    
    def eval(self, y_test, y_pred, labels, target_names):
        """Evaluate classification

        Parameters
        ----------
        y_test : array_like
        y_pred : array_like
        labels : array_like
            encoded class names
        target_names : list
            list with string names of classes     
               
        Returns
        -------
        dict
            dictionary with evaluation metrics, classification report and confusion matrix from sklearn
        """
        results = {}
        # Accuracy
        results['accuracy_score'] = accuracy_score(y_test, y_pred)*100
        # F Score
        results['f_score'] = f1_score(y_test, y_pred, average='weighted')*100
        # Classification report
        results['classification_report'] = classification_report(y_test, y_pred, labels=labels, target_names=target_names, output_dict=True)
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_test, y_pred, labels=labels)
        return results

    def visualize(self, classification_report, confusion_matrix, target_names, output_path, model_name):
        """Visualize classification report and confusion matrix for train_classifier mode

        Based on:
        https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report/58948133#58948133
        
        Parameters
        ----------
        classification_report : dict
        confusion_matrix : ndarray
        target_names : list
            list with string names of classes
        output_path : str
            path to folder to put visualization in
        model_name : str
            name of the model
        """
        heatmap = sns.heatmap(pd.DataFrame(classification_report).iloc[:-1, :].T, annot=True, yticklabels=True, cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True))
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation='horizontal')
        plt.savefig(os.path.join(output_path, 'classification_report_{}.png'.format(model_name)), dpi=300, format='png', bbox_inches='tight')
        plt.close() 
        
        heatmap = sns.heatmap(pd.DataFrame(confusion_matrix).T, annot=True, yticklabels=True, cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True))
        heatmap.set_xticklabels(target_names)
        heatmap.set_yticklabels(target_names, rotation='horizontal')
        plt.savefig(os.path.join(output_path, 'confusion_matrix_{}.png'.format(model_name)), dpi=300, format='png', bbox_inches='tight')
        plt.close() 

    def visualize_pretrained(self, results, output_path, model_name):
        """Visualize classify mode 

        Visualize predicted classes with seaborn catplot 

        Parameters
        ----------
        results : pd DataFrame
        output_path : str
            path to folder to put visualization in
        model_name : str
            name of the model
        """     
        catplot = sns.catplot(data=results, x=model_name, hue='corpus_name', kind='count', palette=sns.color_palette("cubehelix",len(set(results['corpus_name']))))
        catplot._legend.set_title('Corpus Name')
        catplot.set_axis_labels('Class','Count')
        plt.subplots_adjust(bottom=0.1) # necessary to get rid of cut off
        plt.savefig(os.path.join(output_path, 'classes_{}.png'.format(model_name)), dpi=300, format='png')
        plt.close()

    def explain(self, x_train, x_test, feature_names, model_name, output_path):
        """Explain output of model with shap 
        
        Try to explain model with shap
        Catch potential errors if shap not installed or explanation not yet implemented for model

        shap: Lundberg, Scott M and Lee, Su-In. 2019: A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems 30, 4765-4774.
        
        Parameters
        ----------
        x_train : ndarray
        x_test : ndarray
        feature_names : list
        model_name : str
            name of the model
        output_path : str
            path to folder to put visualization in        
        """
        try:
            import shap
            x_train_summary = shap.kmeans(x_train, 10)
            explainer = shap.KernelExplainer(self.model.predict, x_train_summary)
            shap_values = explainer.shap_values(x_test)
            shap.summary_plot(shap_values, x_test, feature_names=feature_names, show=False)
            plt.savefig(os.path.join(output_path, 'shap_summary_{}.png'.format(model_name)), dpi=300, format='png', bbox_inches='tight')
            plt.close()      
        except:
            print('Either you forgot to install shap or visualization of feature importance for {} model has not been implemented.'.format(model_name))

    def scale_split_train(self, x, y, ids):
        """Split x and y in train sets and scale x_train 
        
        Get train split of x, get train split of y, both based on ids.
        If scaler defined for feature package, fit scaler on train split of x of feature package.
        If scaler defined for feature package, scale train split of x of feature package.
        
        Parameters
        ----------
        x : dict of ndarrays
        y : ndarray
        ids : list
        
        Returns
        -------        
        ndarray
        """
        scaled_features = []
        for feature_name, feature_values in x.items():
            x_train = feature_values[ids]
            if feature_name in self._scalers:
                scaled_features.append(self._scalers[feature_name].fit_transform(x_train))
            else:
                scaled_features.append(x_train)
        return np.hstack(tuple(scaled_features)), y[ids]
    
    def scale_split_test(self, x, y, ids):
        """Split x and y in test sets and scale x_test 
        
        Get test split of x, get test split of y, both based on ids.
        If scaler defined for feature package, scale test split of x of feature package with scaler trained on train set.
        
        Parameters
        ----------
        x : dict of ndarrays
        y : ndarray
        ids : list
        
        Returns
        -------        
        ndarray
        """
        scaled_features = []
        for feature_name, feature_values in x.items():
            x_test = feature_values[ids]
            if feature_name in self._scalers:
                scaled_features.append(self._scalers[feature_name].transform(x_test))
            else:
                scaled_features.append(x_test)
        return np.hstack(tuple(scaled_features)), y[ids]  

    def scale_apply(self, x):
        """Scale x (apply set without y)
        If scaler defined for feature package, scale x of feature package with scaler trained on train set.
        
        Parameters
        ----------
        x : dict of ndarrays
        y : ndarray
        ids : list
        
        Returns
        -------        
        ndarray
        """
        scaled_features = []
        for feature_name, feature_values in x.items():
            if feature_name in self._scalers:
                scaled_features.append(self._scalers[feature_name].transform(feature_values))
            else:
                scaled_features.append(feature_values)
        return np.hstack(tuple(scaled_features)) 
    
    def dump(self, model_name, output_path):
        """Dump model and scalers
        
        Parameters
        ----------
        model_name : str
        output_path : str
        """
        self._dump_model(model_name, output_path)
        self._dump_scalers(model_name, output_path)

    def _dump_model(self, model_name, output_path):
        """Dump model 
        
        Parameters
        ----------
        model_name : str
        output_path : str
        """
        pickle.dump(self.model, 
                    open(os.path.join(output_path, model_name+'_model.pickle'), 'wb'))  
       
    def _load_model(self, input_path):
        """Load pretrained model 
        
        Parameters
        ----------
        input_path : str

        Returns
        -------        
        model object
            classification model (sklearn model, etc.)
        """
        return pickle.load(open(input_path+'_model.pickle', 'rb'))       

    def _dump_scalers(self, model_name, output_path):
        """Dump dict of scalers ({feature_package_name:scalers}) for each model
        
        Parameters
        ----------
        model_name : str
        output_path : str
        """
        if self._scalers:
            pickle.dump(self._scalers, 
                        open(os.path.join(output_path, model_name+'_scalers.pickle'), 'wb'))   
        