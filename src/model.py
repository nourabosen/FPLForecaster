import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
import mlflow.keras
import os

def build_model(input_shape, model_type='lstm'):
    model = Sequential()
    if model_type == 'lstm':
        model.add(LSTM(64, activation='tanh', input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32, activation='tanh'))
    elif model_type == 'gru':
        model.add(GRU(64, activation='tanh', input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(32, activation='tanh'))
    
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, model_type='lstm'):
    if X_train is None or len(X_train) == 0:
        return None, None
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        
        model = build_model(input_shape, model_type=model_type)
        model.summary()
        
        callbacks = []
        validation_data = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            callbacks.append(early_stopping)
            validation_data = (X_val, y_val)

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        if validation_data:
            loss, mae = model.evaluate(X_val, y_val, verbose=0)
            mlflow.log_metric("val_loss", loss)
            mlflow.log_metric("val_mae", mae)
        
        mlflow.keras.log_model(model, "model")
        
    return model, history

def predict_next_gameweek(model, player_gw_df, scaler, seq_length=3, feature_cols=['minutes', 'goals', 'assists', 'form', 'fixture_difficulty', 'creativity', 'influence', 'threat']):
    predictions = {}
    sorted_df = player_gw_df.sort_values(['player_id', 'gameweek'])
    
    for player in sorted_df['player_id'].unique():
        player_data = sorted_df[sorted_df['player_id'] == player].reset_index(drop=True)
        if len(player_data) >= seq_length:
            seq_df = player_data.iloc[-seq_length:]
            seq_df[feature_cols] = scaler.transform(seq_df[feature_cols])
            seq = seq_df[feature_cols].values
            seq = np.expand_dims(seq, axis=0)
            pred = model.predict(seq)
            predictions[player] = pred[0, 0]
            
    return predictions

def save_trained_model(model, path="model/fpl_model.h5"):
    """Saves the trained model to the specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_model(model, path)
    print(f"Model saved to {path}")

def load_trained_model(path="model/fpl_model.h5"):
    """Loads a trained model from the specified path."""
    if os.path.exists(path):
        print(f"Loading model from {path}")
        return load_model(path)
    return None

if __name__ == "__main__":
    X_dummy_train = np.random.rand(80, 5, 8)
    y_dummy_train = np.random.rand(80)
    X_dummy_val = np.random.rand(20, 5, 8)
    y_dummy_val = np.random.rand(20)
    
    model, history = train_model(X_dummy_train, y_dummy_train, X_dummy_val, y_dummy_val)
