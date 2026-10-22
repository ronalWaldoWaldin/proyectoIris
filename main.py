"""
Usage:
    main.py train
    main.py compare
    main.py full

Options:
    -h --help
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from docopt import docopt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.utils import to_categorical

# ==================================================
# CONFIGURACION
# ==================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

if not os.path.exists("resultados"):
    os.makedirs("resultados")

# ==================================================
# DATASET
# ==================================================

iris = load_iris()

X = iris.data
y = iris.target

y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=np.argmax(y, axis=1)
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==================================================
# FUNCIONES
# ==================================================

def get_regularizer(reg_type, value):

    if reg_type == "l1":
        return l1(value)

    if reg_type == "l2":
        return l2(value)

    return None


def get_optimizer(name, lr):

    if name == "adam":
        return tf.keras.optimizers.Adam(
            learning_rate=lr
        )

    if name == "sgd":
        return tf.keras.optimizers.SGD(
            learning_rate=lr
        )

    return tf.keras.optimizers.Adam(
        learning_rate=lr
    )


# ==================================================
# MODELO OPTUNA
# ==================================================

def build_model(trial):

    model = Sequential()

    n_layers = trial.suggest_int(
        "n_layers",
        1,
        4
    )

    neurons = trial.suggest_int(
        "neurons",
        8,
        128
    )

    dropout_rate = trial.suggest_float(
        "dropout",
        0.0,
        0.5
    )

    reg_type = trial.suggest_categorical(
        "reg_type",
        [
            "none",
            "l1",
            "l2"
        ]
    )

    reg_value = trial.suggest_float(
        "reg_value",
        1e-6,
        1e-2,
        log=True
    )

    optimizer_name = trial.suggest_categorical(
        "optimizer",
        [
            "adam",
            "sgd"
        ]
    )

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-4,
        1e-2,
        log=True
    )

    regularizer = get_regularizer(
        reg_type,
        reg_value
    )

    for i in range(n_layers):

        if i == 0:

            model.add(
                Dense(
                    neurons,
                    activation="relu",
                    input_shape=(4,),
                    kernel_regularizer=regularizer
                )
            )

        else:

            model.add(
                Dense(
                    neurons,
                    activation="relu",
                    kernel_regularizer=regularizer
                )
            )

        if dropout_rate > 0:
            model.add(
                Dropout(dropout_rate)
            )

    model.add(
        Dense(
            3,
            activation="softmax"
        )
    )

    optimizer = get_optimizer(
        optimizer_name,
        learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==================================================
# OBJETIVO OPTUNA
# ==================================================

def objective(trial):

    model = build_model(trial)

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=100,
        batch_size=16,
        verbose=0
    )

    return max(
        history.history["val_accuracy"]
    )


# ==================================================
# GRAFICOS
# ==================================================

def save_training_graphs(history):

    plt.figure(figsize=(10, 5))

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])

    plt.legend(
        [
            "Train",
            "Validation"
        ]
    )

    plt.title("Loss")

    plt.savefig(
        "resultados/loss.png"
    )

    plt.close()

    plt.figure(figsize=(10, 5))

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])

    plt.legend(
        [
            "Train",
            "Validation"
        ]
    )

    plt.title("Accuracy")

    plt.savefig(
        "resultados/accuracy.png"
    )

    plt.close()


# ==================================================
# MATRIZ DE CONFUSION
# ==================================================

def save_confusion_matrix(model):

    y_pred = model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(
        y_pred,
        axis=1
    )

    y_true = np.argmax(
        y_test,
        axis=1
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(6, 6))

    plt.imshow(
        cm,
        cmap="Blues"
    )

    plt.colorbar()

    plt.title(
        "Confusion Matrix"
    )

    plt.savefig(
        "resultados/confusion_matrix.png"
    )

    plt.close()


# ==================================================
# COMPARAR REGULARIZACIONES
# ==================================================

def compare_regularizations():

    methods = [
        "none",
        "l1",
        "l2",
        "dropout"
    ]

    accuracies = []

    for method in methods:

        model = Sequential()

        if method == "l1":

            model.add(
                Dense(
                    64,
                    activation="relu",
                    input_shape=(4,),
                    kernel_regularizer=l1(0.001)
                )
            )

        elif method == "l2":

            model.add(
                Dense(
                    64,
                    activation="relu",
                    input_shape=(4,),
                    kernel_regularizer=l2(0.001)
                )
            )

        else:

            model.add(
                Dense(
                    64,
                    activation="relu",
                    input_shape=(4,)
                )
            )

        if method == "dropout":

            model.add(
                Dropout(0.3)
            )

        model.add(
            Dense(
                3,
                activation="softmax"
            )
        )

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.fit(
            X_train,
            y_train,
            epochs=100,
            verbose=0
        )

        pred = model.predict(
            X_test,
            verbose=0
        )

        pred = np.argmax(
            pred,
            axis=1
        )

        acc = accuracy_score(
            np.argmax(
                y_test,
                axis=1
            ),
            pred
        )

        accuracies.append(acc)

    plt.figure(figsize=(8, 5))

    plt.bar(
        methods,
        accuracies
    )

    plt.title(
        "Comparacion Regularizaciones"
    )

    plt.savefig(
        "resultados/comparacion_regularizaciones.png"
    )

    plt.close()


# ==================================================
# COMPARAR OPTIMIZADORES
# ==================================================

def compare_optimizers():

    optimizers = [
        "adam",
        "sgd"
    ]

    accuracies = []

    for opt in optimizers:

        model = Sequential([
            Dense(
                64,
                activation="relu",
                input_shape=(4,)
            ),
            Dense(
                3,
                activation="softmax"
            )
        ])

        model.compile(
            optimizer=opt,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.fit(
            X_train,
            y_train,
            epochs=100,
            verbose=0
        )

        pred = model.predict(
            X_test,
            verbose=0
        )

        pred = np.argmax(
            pred,
            axis=1
        )

        acc = accuracy_score(
            np.argmax(
                y_test,
                axis=1
            ),
            pred
        )

        accuracies.append(acc)

    plt.figure(figsize=(8, 5))

    plt.bar(
        optimizers,
        accuracies
    )

    plt.title(
        "Comparacion Optimizadores"
    )

    plt.savefig(
        "resultados/comparacion_optimizadores.png"
    )

    plt.close()


# ==================================================
# ENTRENAMIENTO
# ==================================================

def train():

    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        objective,
        n_trials=40
    )

    with open(
        "resultados/best_params.json",
        "w"
    ) as f:

        json.dump(
            study.best_params,
            f,
            indent=4
        )

    best_trial = optuna.trial.FixedTrial(
        study.best_params
    )

    model = build_model(
        best_trial
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=150,
        batch_size=16,
        verbose=1
    )

    model.save(
        "resultados/mejor_modelo.keras"
    )

    save_training_graphs(
        history
    )

    save_confusion_matrix(
        model
    )

    pred = model.predict(
        X_test,
        verbose=0
    )

    pred = np.argmax(
        pred,
        axis=1
    )

    acc = accuracy_score(
        np.argmax(
            y_test,
            axis=1
        ),
        pred
    )

    pd.DataFrame(
        {
            "accuracy": [acc]
        }
    ).to_csv(
        "resultados/metricas.csv",
        index=False
    )

    print(
        classification_report(
            np.argmax(
                y_test,
                axis=1
            ),
            pred
        )
    )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    args = docopt(__doc__)

    if args["train"]:
        train()

    elif args["compare"]:
        compare_regularizations()
        compare_optimizers()

    elif args["full"]:
        compare_regularizations()
        compare_optimizers()
        train()