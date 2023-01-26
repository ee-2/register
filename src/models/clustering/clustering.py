import os
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import seaborn as sns
from utils.diverse_helpers import LegendTitle

class Clustering():
    """Clustering super class
    
    Only used for inheritance of functions to different clustering models

    See Also
    --------
    models.clustering.clustering_sklearn
    """

    def fit(self, x):
        """Fit model 
        
        Parameters
        ----------
        x : ndarray
        """
        self.model.fit(x)
        
    def fit_predict(self, x):
        """Fit and transform x
        
        Parameters
        ----------
        x : ndarray
                    
        Returns
        -------
        ndarray
        """
        return self.model.fit_predict(x)
    
    def eval(self, y_true, y_pred):
        """Evaluate clustering
        
        Evaluate clustering in cases where classes are given

        Parameters
        ----------
        y_test : array_like
        y_pred : array_like
        
        Returns
        -------
        dict
            dictionary with evaluation metrics
        """
        return {'ARI':adjusted_rand_score(y_true, y_pred)}
    
    def visualize(self, x_scaled, results, corpora_file_ids, model_name, output_path):
        """Visualize clustering results
        
        Visualize cluster mode with a seaborn scatter plot.
        For agglomerative clustering a dendrogram is plotted too.
        
        Visualization of Dendrogram from https://github.com/scikit-learn/scikit-learn/blob/70cf4a676caa2d2dad2e3f6e4478d64bcb0506f7/examples/cluster/plot_hierarchical_clustering_dendrogram.py

        Parameters
        ----------
        x_scaled : ndarray
        results : pd DataFrame
        corpora_file_ids : dict
            dictionary with dictionary for each corpus with file as key and index as value
        output_path : str
            path to folder to put visualization in
        model_name : str
            name of the model
        """
        x_2d = self._lin_dim_reduction(x_scaled)
        
        scatterplot = sns.scatterplot(x=x_2d[:, 0], y=x_2d[:, 1], hue=['Cluster {}'.format(label) for label in self.model.labels_], style=results['corpus_name'], palette=sns.color_palette("cubehelix",len(set(self.model.labels_))))
        
        if len(corpora_file_ids) == 1:
            for line in range(0,x_2d.shape[0]):
                scatterplot.text(x_2d[line][0], x_2d[line][1], results.index[line], horizontalalignment='left', color='black', fontsize='6')
        
        legend_markers, legend_labels = scatterplot.get_legend_handles_labels() # wait for subtitle handling
        scatterplot.legend(['Cluster']+legend_markers[:len(set(self.model.labels_))]+['Corpus']+legend_markers[len(set(self.model.labels_)):], ['']+legend_labels[:len(set(self.model.labels_))]+['']+legend_labels[len(set(self.model.labels_)):], handler_map={str: LegendTitle()})
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.savefig(os.path.join(output_path, 'pca_{}.png'.format(model_name)), dpi=300, format='png')
        plt.close()
        
        if model_name == 'AgglomerativeClustering':
            children = self.model.children_
            distance = np.arange(children.shape[0])
            no_of_observations = np.arange(2, children.shape[0]+2)
            linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
            dendrogram(linkage_matrix, orientation='right', labels=['{} {}'.format(corpus, file) for corpus, files in corpora_file_ids.items() for file in files])    
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'dendrogram_{}.png'.format(model_name)), dpi=300, format='png')     
            plt.close()     

    def explain(self, x, feature_names, model_name, output_path):
        """Explain output of model with shap 
        
        Try to explain model with shap
        Catch potential errors if shap not installed or explanation not yet implemented for model

        shap: Lundberg, Scott M and Lee, Su-In. 2019: A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems 30, 4765-4774.
        
        Parameters
        ----------
        x : ndarray
        feature_names : list
        model_name : str
            name of the model
        output_path : str
            path to folder to put visualization in        
        """
        try:
            import shap
            explainer = shap.KernelExplainer(self.model.predict, x)
            shap_values = explainer.shap_values(x)
            shap.summary_plot(shap_values, x, feature_names=feature_names, show=False)
            plt.savefig(os.path.join(output_path, 'shap_summary_{}.png'.format(model_name)), dpi=300, format='png', bbox_inches='tight')
            plt.close()    
        except:
            print('Either you forgot to install shap or visualization of feature importance for {} model has not been implemented.'.format(model_name))
             
    def scale(self, x):
        """Scale x
        
        If scaler defined for feature package, fit scaler on x of feature package.
        If scaler defined for feature package, scale x of feature package.
        
        Parameters
        ----------
        x : dict of ndarrays
        
        Returns
        -------        
        ndarray
        """
        scaled_features = []
        for feature_name, feature_values in x.items():
            if feature_name in self._scalers:
                scaled_features.append(self._scalers[feature_name].fit_transform(feature_values))
            else:
                scaled_features.append(feature_values)
        return np.hstack(tuple(scaled_features))

    def _lin_dim_reduction(self, x_scaled):
        """Perform PCA to plot clustering
        
        Parameters
        ----------
        x_scaled : ndarray
        
        Returns
        -------        
        ndarray
        """
        return PCA(n_components=2).fit_transform(x_scaled)
