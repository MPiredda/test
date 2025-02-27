import joblib
import numpy as np
def load_models():
    return {
        "mlp": joblib.load('mlp_model.joblib'),
        "rf": joblib.load('rf_model.joblib'),
        "dnn": joblib.load('dnn_model.joblib'),
        "xgb": joblib.load('xgb_model.joblib'),
        "scaler_mlp": joblib.load('scaler.joblib'),
        "scaler_dnn": joblib.load('dnn_scaler.joblib')
    }


def predict_password_strength(models, sample):
    # Normalizzare i dati per MLP e DNN
    sample_scaled_mlp = models["scaler_mlp"].transform(sample)
    sample_scaled_dnn = models["scaler_dnn"].transform(sample)

    predictions = {
        "MLP": models["mlp"].predict(sample_scaled_mlp)[0],
        "Random Forest": models["rf"].predict(sample)[0],
        "DNN": models["dnn"].predict(sample_scaled_dnn)[0],
        "XGBoost": models["xgb"].predict(sample)[0],
    }

    return predictions


# Eseguire la previsione su un nuovo esempio
models = load_models()
new_sample = np.array([[4, 5, 2]])  # Digit Count, Lowercase Count, Uppercase Count
predictions = predict_password_strength(models, new_sample)
print(predictions)
