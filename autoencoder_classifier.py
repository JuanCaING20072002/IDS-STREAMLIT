# autoencoderclassifier.py

from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import numpy as np

class AutoencoderClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=None, n_classes=2, epochs=100, batch_size=32, verbose=0, patience=5):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.patience = patience
        self.model = None

    def _build_model(self):
        input_layer = Input(shape=(self.input_dim,))
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dense(32, activation='relu')(encoded)
        latent  = Dense(16, activation='relu', name='latent_space')(encoded)

        decoded = Dense(32, activation='relu')(latent)
        decoded = Dense(64, activation='relu')(decoded)
        reconstruction = Dense(self.input_dim, activation='sigmoid', name='reconstruction')(decoded)

        classifier = Dense(32, activation='relu')(latent)
        classifier = Dense(self.n_classes, activation='softmax', name='classification')(classifier)

        model = Model(inputs=input_layer, outputs=[reconstruction, classifier])
        model.compile(optimizer=Adam(),
                      loss={'reconstruction': 'mse', 'classification': 'categorical_crossentropy'},
                      loss_weights={'reconstruction': 0.5, 'classification': 1.0},
                      metrics={'classification': 'accuracy'})
        return model

    def fit(self, X, y):
        self.input_dim = X.shape[1]
        self.n_classes = len(set(y)) if len(y.shape) == 1 else y.shape[1]
        y_cat = to_categorical(y, num_classes=self.n_classes)
        self.model = self._build_model()

        early_stop = EarlyStopping(
            monitor='val_classification_accuracy',
            mode='max',  # Esto evita el error anterior
            patience=self.patience,
            restore_best_weights=True,
            verbose=self.verbose > 0
        )

        self.model.fit(X, {'reconstruction': X, 'classification': y_cat},
                       validation_split=0.2,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=self.verbose,
                       callbacks=[early_stop])
        return self

    def predict(self, X):
        _, y_pred = self.model.predict(X)
        return np.argmax(y_pred, axis=1)

    def predict_proba(self, X):
        _, y_pred = self.model.predict(X)
        return y_pred

    def reconstruction_error(self, X):
        """Return per-sample mean squared reconstruction error.
        Uses the autoencoder's reconstruction output compared against X.
        """
        y_rec, _ = self.model.predict(X)
        # Ensure shapes align
        if y_rec.shape != X.shape:
            # Try to coerce if possible; else fallback to zero error
            try:
                X_arr = np.asarray(X)
                errs = np.mean((X_arr - y_rec)**2, axis=1)
            except Exception:
                errs = np.zeros(len(X))
            return errs
        return np.mean((X - y_rec)**2, axis=1)
