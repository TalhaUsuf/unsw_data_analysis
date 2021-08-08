import joblib
from django.apps import AppConfig
from django.conf import settings
import os
import tensorflow as tf
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from rich.console import Console
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from rich.table import Table
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import backend as K

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                      what to do when app loads
#      load the model
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class MLModelConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ML_model'

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess_pipeline = ColumnTransformer(transformers=[
        ("num", numeric_transformer, selector(dtype_exclude=object)),
        ("cat", categorical_transformer, selector(dtype_include=object))
    ])
    inp = tf.keras.Input(shape=(205))
    x = tf.keras.layers.Dense(10, activation=tf.nn.relu)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)

    model = tf.keras.Model(inputs=[inp], outputs=[out])
    model.load_weights(settings.MODELS)
    preprocess = joblib.load((settings.PREPROCESSOR))
    print("models loaded ....")



