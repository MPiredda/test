import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_curve, auc, make_scorer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Caricamento del dataset dal file CSV
df = pd.read_csv('dataset_feacture.csv')

# Verifica della struttura del dataset
print(df.head())


# Estrazione delle feature e della label
X = df[['Digit Count', 'Lowercase Count', 'Uppercase Count', 'Special Char Count', 'Unique Char Count', 'Total Length']]
y = df['strength']

# Normalizzazione delle feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Suddivisione del dataset in 70% train, 20% test e 10% validation
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)  # Circa 20% test e 10% validation

# Definizione del modello MLP
mlp = MLPClassifier(max_iter=1000, random_state=42)

# Definizione della griglia di iperparametri da ottimizzare
param_grid = {
    'hidden_layer_sizes': [(10,10), (50,50), (100,50)],  # Diverse configurazioni dei neuroni
    'activation': ['relu', 'tanh'],                     # Funzioni di attivazione
    'solver': ['adam', 'sgd'],                          # Algoritmi di ottimizzazione
    'alpha': [0.0001, 0.001],                           # Regularizzazione L2 (parametro di penalità)
    'learning_rate': ['constant', 'adaptive']           # Strategia di apprendimento
}

# Configurazione di GridSearchCV con 5-fold cross-validation
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Addestramento con la ricerca degli iperparametri
grid_search.fit(X_train, y_train)

# Visualizzazione dei migliori iperparametri
print("Migliori iperparametri trovati:", grid_search.best_params_)


# Miglior modello trovato
best_mlp = grid_search.best_estimator_

# Cross-validation a 5-folds per valutare le prestazioni
cv_scores = cross_val_score(best_mlp, X_train, y_train, cv=5, scoring='accuracy')
print("5-fold Cross-validation accuracy: ", np.mean(cv_scores))

# Valutazione finale sul set di validazione
y_val_pred = best_mlp.predict(X_val)
print("Valutazione sul set di validazione:")
print(classification_report(y_val, y_val_pred))


# Calcolo delle probabilità per ogni classe
y_test_proba = best_mlp.predict_proba(X_test)

# Calcolo della ROC curve e dell'AUC per la classe 1 (forza debole)
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)

# Disegno della ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
