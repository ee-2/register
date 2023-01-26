from tensorflow import keras
from .linear_regression import LinearRegression
import os
from sklearn.model_selection import train_test_split
import numpy as np

class LinearRegression_tf_keras(LinearRegression):
    """Simple keras (tensorflow) model for linear regression

    Inherit base functions from models.linearregression.linear_regression.LinearRegression
    
    If you want other options for your model, build new model in the same style and 
    set the model's lib parameter in the config to the part of the name of the class after the first _. 
    
    Parameters
    ----------    
    dim : int
        dimension of features input
    scalers : dict 
        dictionary with feature packages and respective scalers
    model : str, default='NeuralNetwork'
        name of model
    hidden_layer_dimensions: list of int, default=[128,64])
        list with dimensions of hidden layers, first entry is dimension of first hidden layer etc.
        applies only if nr_hidden_layers is not set (0, default)
    nr_hidden_layers: int, default=0
        if nr_hidden_layers specified, model with respective number of hidden layers is build, 
        each dimension of a hidden layer is half the preceeding hidden layer (breaks if dim < 2)
    dropout_input: float, default=0.2
        dropout rate to use on the input layer
    dropout_hidden: float, default=0.5
        dropout rate to use on the hidden layers
    path : None or str, optional
        path to pretrained model
        default is None (model will be trained)
    config : dict, optional
        other configuration parameters, will not be used for tf_keras model (just for implementation reasons)
    
    Attributes
    ----------
    model : model-object
        sklearn model
    _scalers : dict
        dictionary with feature packages and respective scalers
    _callbacks : list
        keras callbacks

    Notes
    -----
    define in config file under models via:    
    "models":[    
                {  
                    "lib": "tf_keras",  
                    "model": "NeuralLinearRegressor",
                    "hidden_layer_dimensions": [256, 128],
                    
                }
            ]
    """
    def __init__(self, dim, scalers, model='NeuralNetwork', hidden_layer_dimensions=[128,64], nr_hidden_layers=0, dropout_input=0.2, dropout_hidden=0.5, path=None, **config):
        if path:
            self.model = self._load_model(path)
        else: 
            self.model = keras.models.Sequential(name=model)
            # input layer
            self.model.add(keras.layers.Dropout(dropout_input, input_shape=(dim,)))        
            
            # hidden layers:           
            if nr_hidden_layers: # if nr_hidden_layers specified (each dimension is half the preceeding one (breaks if dim < 2)
                hidden_layer_dimensions = []
                for i in range(nr_hidden_layers):
                    dim = int(dim/2) # mean?
                    if dim <2:
                        break   
                    hidden_layer_dimensions.append(dim)        
                           
            for dim in hidden_layer_dimensions:
                self.model.add(keras.layers.Dense(dim, kernel_initializer='random_normal', bias_initializer='random_normal', kernel_constraint=keras.constraints.MaxNorm(3)))
                self.model.add(keras.layers.Dropout(dropout_hidden))
                self.model.add(keras.layers.Activation('relu'))
    
            # output layer
            self.model.add(keras.layers.Dense(1, activation='linear', kernel_initializer='random_normal', bias_initializer='random_normal'))                           
            
            self.model.compile(loss='mse', optimizer='adam', metrics=['mae','mse'])
            self.model.summary()
            
            self._callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)]
        self._scalers = scalers

    def fit(self, x_train, y_train):
        """Fit model 
        
        Fit model with 1000 epochs, batch_size of 32 and early stopping
        
        Parameters
        ----------
        x_train : tuple of ndarrays
            contains train and dev split of x (x_train-ndarray, x_dev-ndarray)
        y_train : tuple of ndarray
            contains train and dev split of y (y_train-ndarray, y_dev-ndarray)
        """
        x_train, x_dev = x_train 
        y_train, y_dev = y_train
        history = self.model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_data=(x_dev, y_dev), callbacks=self._callbacks)
    
    def predict(self, x_test):
        """Predict scores 
        
        Workaround to avoid retracing: https://github.com/tensorflow/tensorflow/issues/42441
        
        Parameters
        ----------
        x_test : ndarray
                    
        Returns
        -------
        array_like
        """
        return self.model.predict_step(x_test).numpy().ravel()

    def explain(self, x_train, x_test, feature_names, model_name, output_path):
        """Explain output of model with shap 
        
        For DeepExplainer: wait for shap-update for tensorflow2 compatibility https://github.com/slundberg/shap/pull/1483
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
            import matplotlib.pyplot as plt
            x_train, x_dev = x_train 
            x_train_summary = shap.kmeans(x_train, 10)
            explainer = shap.KernelExplainer(self.model.predict, x_train_summary)
            shap_values = explainer.shap_values(x_test)
            shap.summary_plot(shap_values[0], x_test, feature_names=feature_names, show=False)
            plt.savefig(os.path.join(output_path, 'shap_summary_{}.png'.format(model_name)), dpi=300, format='png', bbox_inches='tight')
            plt.close()
        except:
            print('Either you forgot to install shap or visualization of feature importance for {} model has not been implemented.'.format(model_name))    

    def scale_split_train(self, x, y, ids):
        """Split x and y in train and dev sets and scale x_train and x_dev
        
        Split train ids into train and dev ids to fit scalers only on train.

        Parameters
        ----------
        x : dict of ndarrays
        y : ndarray
        ids : list
        
        Returns
        -------        
        tuple of ndarrays
        tuple of ndarrays
        """
        train_ids, dev_ids = train_test_split(ids, test_size=0.2, random_state=42)
        x_train, y_train = self._scale_split_train(x, y, train_ids)
        x_dev, y_dev = self.scale_split_test(x, y, dev_ids)
        return (x_train, x_dev), (y_train, y_dev)

    def _scale_split_train(self, x, y, ids):
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

    def _dump_model(self, model_name, output_path):
        """Dump model 
        
        Parameters
        ----------
        model_name : str
        output_path : str
        """
        self.model.save(os.path.join(output_path, model_name+'_model'))  
        
    def _load_model(self, input_path):
        """Load pretrained model 
        
        Parameters
        ----------
        input_path : str

        Returns
        -------        
        model object
            Tensorflow-keras model
        """
        return keras.models.load_model(input_path+'_model')

        