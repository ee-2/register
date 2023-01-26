import scipy.stats as st
from sklearn.metrics import mean_squared_error, r2_score
import pickle, os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class LinearRegression():
    """Linear Regression super class
    
    Only used for inheritance of functions to different linear regression models (sklearn or tf_keras)

    See Also
    --------
    models.linearregression.linear_regression_tf_keras
    models.linearregression.linear_regression_sklearn
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
        """Predict scores 
        
        Parameters
        ----------
        x_test : ndarray
                    
        Returns
        -------
        array_like
        """
        return self.model.predict(x_test)
    
    def eval(self, y_test, y_pred):
        """Evaluate linear regression 

        Parameters
        ----------
        y_test : array_like
        y_pred : array_like
        
        Returns
        -------
        dict
            dictionary with evaluation metrics
        """
        pearson_r, pearson_p = st.pearsonr(y_test, y_pred)
        spearman_r, spearman_p = st.spearmanr(y_test, y_pred)
        kendall_tau, kendall_p = st.kendalltau(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        return {'pearson_r':pearson_r, 'pearson_p':pearson_p, 
                'spearman_r':spearman_r, 'spearman_p':spearman_p, 
                'kendall_tau':kendall_tau, 'kendall_p':kendall_p,
                'r2_score':r2, 
                'mean_squared_error':mse}

    def visualize(self, y_true, y_pred, output_path, model_name):
        """Visualize true vs. predicted values for train_linear_regressor mode

        Parameters
        ----------
        y_test : array_like
        y_pred : array_like
        output_path : str
            path to folder to put visualization in
        model_name : str
            name of the model
        """    
        sns.histplot({'True':y_true, 'Predicted':y_pred}, x='True', y='Predicted', cbar=True)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=0.5)
        plt.savefig(os.path.join(output_path, 'true_pred_{}.png'.format(model_name)), dpi=300, format='png', bbox_inches='tight')
        plt.close()      
 
    def visualize_pretrained(self, results, output_path, model_name):
        """Visualize score mode
        
        Visualize predicted scores with seaborn displot 

        Parameters
        ----------
        results : pd DataFrame
        output_path : str
            path to folder to put visualization in
        model_name : str
            name of the model
        """    
        displot = sns.displot(results, x=model_name, hue='corpus_name', element='step', palette=sns.color_palette("cubehelix",len(set(results['corpus_name']))))
        displot._legend.set_title('Corpus Name')
        displot.set_xlabels('Score')
        plt.savefig(os.path.join(output_path, 'scores_{}.png'.format(model_name)), dpi=300, format='png', bbox_inches='tight')
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
            LinearRegression model (sklearn model, etc.)
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
        