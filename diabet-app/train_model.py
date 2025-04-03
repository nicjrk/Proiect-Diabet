# 1. Importuri
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# 2. Setare seed pentru reproducibilitate
def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

# 3. Încărcare și preprocesare date
def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    X = df.drop('class', axis=1).copy()
    y = df['class'].map({'Negative': 0, 'Positive': 1})

    # Codificare binară explicită
    for col in X.columns:
        unique_vals = set(X[col].unique())
        if unique_vals == {'Yes', 'No'}:
            X[col] = X[col].map({'No': 0, 'Yes': 1})
        elif col == 'Gender':
            X[col] = X[col].map({'Female': 0, 'Male': 1})

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# 4. Construirea modelului
def build_model(input_dim):
    model = Sequential([
        tf.keras.Input(shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# 5. Antrenare model cu EarlyStopping
def train_model(X, y, seed=42):
    set_seeds(seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, shuffle=True)

    model = build_model(X_train.shape[1])
    early_stop = EarlyStopping(monitor='val_accuracy', patience=20, 
                               restore_best_weights=True, mode='max')

    history = model.fit(X_train, y_train, validation_split=0.2, epochs=300,
                        batch_size=16, callbacks=[early_stop], verbose=0)

    y_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_proba > 0.5).astype("int32")
    test_acc = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)

    return model, history, X_test, y_test, y_pred, y_proba, test_acc, auc_score

# 6. Vizualizare performanță
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title("Acuratețe")
    plt.xlabel("Epoci")
    plt.ylabel("Acuratețe")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Pierdere")
    plt.xlabel("Epoci")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 7. Matrice de confuzie și clasificare
def evaluate_model(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matrice de confuzie")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    print("\n📄 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# 8. Afișare probabilități predicție
def print_prediction_confidences(y_proba, y_pred):
    print("\n🔍 Probabilități pentru fiecare exemplu din setul de test:")
    for i, (prob, pred) in enumerate(zip(y_proba, y_pred)):
        label = "Positive" if pred == 1 else "Negative"
        print(f"Exemplul {i+1}: {label} ({prob:.2%} certitudine)")

# 9. Executare completă
if __name__ == '__main__':
    X_scaled, y = load_and_preprocess_data('diabetes_data_upload.csv')
    model, history, X_test, y_test, y_pred, y_proba, acc, auc = train_model(X_scaled, y)

    print(f"\n✅ Acuratețe pe test: {acc:.4f}")
    print(f"🎯 AUC Score: {auc:.4f}")

    plot_history(history)
    evaluate_model(y_test, y_pred)
    print_prediction_confidences(y_proba, y_pred)

    model.save("best_diabetes_model.keras")
