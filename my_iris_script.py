# Importer les bibliothèques nécessaires
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Charger le jeu de données Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Diviser le jeu de données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Normaliser les données (important pour les algorithmes basés sur la distance comme k-NN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Créer un classificateur k-NN avec k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Prédire sur l'ensemble de test
y_pred = knn.predict(X_test)

# Évaluer le modèle
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))
print("\nExactitude :")
print(accuracy_score(y_test, y_pred))
