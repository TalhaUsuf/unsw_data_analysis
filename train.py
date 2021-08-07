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
import joblib

def encoder_cat_data(cat_cols : list, df : pd.DataFrame, is_train : bool):
    global cat_enc
    if is_train:
        Console().rule(f"transforming cat. labels of [yellow]train[/yellow] data ... ", align="left")
        Console().print(df[cat_cols].head(10))
        encoded_features = df[cat_cols].apply(cat_enc.fit_transform)
        Console().print(f"classes are ---> {cat_enc.classes_}")
        return encoded_features
    elif not is_train:
        Console().rule(f"transforming cat. labels of [blue]test[/blue] data ... ", align="left")
        encoded_features = df[cat_cols].apply(cat_enc.transform)
        return encoded_features


def scaler_num_data(num_cols : list, df : pd.DataFrame, is_train : bool):
    global num_enc
    if is_train:
        Console().rule(f"transforming num. data of [yellow]train[/yellow] data ... ", align="left")
        encoded_features = df[num_cols].apply(num_enc.fit_transform)
        return encoded_features
    elif not is_train:
        Console().rule(f"transforming num. data of [blue]test[/blue] data ... ", align="left")
        encoded_features = df[num_cols].apply(num_enc.transform)
        return encoded_features



def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def main():
    with Console().status(f"PRE PROCESSING .... ", spinner="bouncingBall"):
        # paths
        test = Path("UNSW_NB15_testing-set.csv")
        train = Path("UNSW_NB15_training-set.csv")

        # separate cataegorical columns
        df_test = pd.read_csv(test.as_posix(), skipinitialspace=True)
        df_train = pd.read_csv(train.as_posix(), skipinitialspace=True)

        Console().rule(f"[yellow]stats of the training dataset", style="red on green")
        Console().print(df_train.info(null_counts=True))

        Console().rule(f"[yellow]stats of the testing dataset", style="red on green")
        Console().print(df_test.info(null_counts=True))

        numeric_cols_train = df_train._get_numeric_data().columns
        numeric_cols_test = df_train._get_numeric_data().columns

        cat_cols_train = list(set(df_train.columns) - set(numeric_cols_train))
        cat_cols_test = list(set(df_test.columns) - set(numeric_cols_test))

        assert cat_cols_test==cat_cols_train , "categorical columns should be same in train and test data"

        tab = Table(title=f"Non. numerical columns", title_style="yellow bold", style="cyan", show_lines=True)
        tab.add_column("[red]In Train Data", style="magenta")
        tab.add_column("[red]In Test Data", style="green")
        for cat_train, cat_test in zip(cat_cols_train, cat_cols_test):
            tab.add_row(f"{cat_train}", f"{cat_test}")

        Console().print(tab)

    with Console().status(f"Combining data .... ", spinner="bouncingBall"):


        categorical_data = pd.concat([df_train[cat_cols_train], df_test[cat_cols_test]], axis=0)
        numerical_data = pd.concat([df_train[numeric_cols_train], df_test[numeric_cols_test]], axis=0)

        Console().rule(characters='%')
        Console().print(f"total categorical data shape ----> {categorical_data.shape}")
        Console().print(f"total numerical data shape ----> {numerical_data.shape}")
        Console().rule(characters='%')

        Console().print("categorical data has ----> ",categorical_data.isna().sum().sum(), "nan values")
        Console().print("numerical data has ----> ",numerical_data.isna().sum().sum(), "nan values")

    with Console().status(f"Constructing preprocessing pipelines ....", spinner="bouncingBall"):

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

        total_df = pd.concat([categorical_data, numerical_data], axis=1)

        Console().print(f"Combined dataframe shape ===> {total_df.shape}")

        traindf, testdf = train_test_split(total_df, test_size=0.30, shuffle=True, stratify=total_df["label"])
        xtrain = traindf.drop(["label"],axis=1)
        ytrain = traindf[["label"]]
        xtest = testdf.drop(["label"],axis=1)
        ytest = testdf[["label"]]

        Console().print("\n")
        Console().rule(characters="%")
        Console().print(f"train-frame-dataset shape :::::::>> {xtrain.shape}")
        Console().print(f"train-labels-dataset shape :::::::>> {ytrain.shape}")
        Console().print(f"test-frame-dataset shape :::::::>> {xtest.shape}")
        Console().print(f"test-labels-dataset shape :::::::>> {ytest.shape}")
        Console().rule(characters="%")
        pd.DataFrame.from_dict({"columns":xtrain.columns.tolist()}).to_csv(f"./column_inputs.csv", index=False)

        Console().rule(f"\n\ninput columns needed at test-time are stored in csv file {'./column_inputs.csv'} ", style="red on yellow", align="center")
        del df_test, df_train, traindf, testdf
        Console().print("\n")
        # preprocess_pipeline.fit_transform()
        Console().print(xtrain.info())
        xtrain = preprocess_pipeline.fit_transform(xtrain)
        xtest = preprocess_pipeline.transform(xtest)
        Console().status("successfully applied the pre-processing pipeline")
        #
        # xtrain = xtrain.values
        ytrain = ytrain.values
        # xtest = xtest.values
        ytest = ytest.values
        # Console().print(f"labels are ===> {ytest}")



    with Console().status("training ... ",spinner="aesthetic"):
        inp = tf.keras.Input(shape=(xtrain.shape[-1]))
        x = tf.keras.layers.Dense(10, activation=tf.nn.relu)(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)

        model = tf.keras.Model(inputs=[inp], outputs=[out])

        Console().print(model.summary())
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

        checkpoint_filepath = 'weights.{epoch:02d}-{val_f1_m:.2f}.hdf5'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_f1_m',
            mode='max',
            save_best_only=True)

        history = model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=3, verbose=1, batch_size=64, callbacks=[model_checkpoint_callback])
        joblib.dump(history, f"history.pkl")







if __name__=='__main__':
    main()