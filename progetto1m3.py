
# ============================================================
# PROGETTO 1 - CLUSTERING SU DATASET IRIS
# ============================================================
# Obiettivi:
# 1. Esplorare il dataset Iris.
# 2. Standardizzare le caratteristiche.
# 3. Applicare DBSCAN con diversi valori di eps.
# 4. Utilizzare Elbow Method per scegliere K.
# 5. Addestrare K-Means con il numero di cluster scelto.
# 6. Calcolare la Silhouette Score.
# 7. Visualizzare cluster e centroidi.
#
# NOTA:
# Utilizziamo soltanto le caratteristiche X.
# Non utilizziamo le etichette reali delle specie (y),
# perché stiamo svolgendo un problema di clustering
# non supervisionato.
# ============================================================


# ------------------------------------------------------------
# 1. IMPORTAZIONE DELLE LIBRERIE
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score


# ------------------------------------------------------------
# 2. CARICAMENTO ED ESPLORAZIONE DEL DATASET
# ------------------------------------------------------------

# Carichiamo il dataset Iris
iris = load_iris()

# Estraiamo esclusivamente le quattro caratteristiche
# dei fiori, senza utilizzare le etichette delle specie.
X = iris.data

# Creiamo un DataFrame per esplorare i dati
df = pd.DataFrame(
    X,
    columns=iris.feature_names
)

# Visualizziamo statistiche descrittive:
# conteggio, media, deviazione standard, minimo,
# quartili e massimo di ogni caratteristica.
print(df.describe())


# ------------------------------------------------------------
# 3. STANDARDIZZAZIONE DEI DATI
# ------------------------------------------------------------

# Creiamo uno StandardScaler.
# StandardScaler trasforma le caratteristiche in modo
# che abbiano media circa 0 e deviazione standard circa 1.
scaler = StandardScaler()

# Applichiamo la standardizzazione a tutte le caratteristiche.
X_std = scaler.fit_transform(X)

# Controlliamo che la standardizzazione sia avvenuta correttamente.
print("\nMedie standardizzate:")
print(X_std.mean(axis=0))

print("\nDeviazioni standard:")
print(X_std.std(axis=0))


# ------------------------------------------------------------
# 4. DBSCAN CON DIVERSI VALORI DI EPS
# ------------------------------------------------------------

# Proviamo tre diversi valori di eps.
# EPS rappresenta il raggio di vicinanza utilizzato
# da DBSCAN per individuare i punti vicini.
eps_values = [0.35, 0.45, 0.55]

# Numero minimo di punti richiesti nella regione
# di vicinanza per identificare una zona densa.
min_samples = 5

# Proviamo DBSCAN separatamente per ogni valore di eps.
for eps in eps_values:

    # Creiamo il modello DBSCAN con i parametri scelti.
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    # Addestriamo il modello e otteniamo le etichette
    # assegnate ai punti.
    # Il valore -1 identifica i punti considerati rumore.
    labels_dbscan = dbscan.fit_predict(X_std)

    # Contiamo i cluster individuati.
    # Escludiamo -1 perché non rappresenta un cluster,
    # ma identifica i punti rumorosi.
    n_clusters = (
        len(set(labels_dbscan))
        - (1 if -1 in labels_dbscan else 0)
    )

    # Contiamo quanti punti sono stati classificati come rumore.
    noise = np.sum(labels_dbscan == -1)

    # Stampiamo i risultati per ogni configurazione.
    print(f"\neps: {eps}")
    print(f"Cluster individuati: {n_clusters}")
    print(f"Punti rumorosi: {noise}")


# ------------------------------------------------------------
# 5. ELBOW METHOD - RICERCA DEL NUMERO DI CLUSTER
# ------------------------------------------------------------

# Lista vuota che conterrà i valori di WCSS
# (Within-Cluster Sum of Squares).
wcss = []

# Valori di K da testare: da 1 a 10 cluster.
k_values = range(1, 11)

# Addestriamo un K-Means per ogni valore di K.
for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    # Addestriamo il modello sui dati standardizzati.
    kmeans.fit(X_std)

    # inertia_ restituisce la WCSS:
    # la somma delle distanze al quadrato
    # tra i punti e i rispettivi centroidi.
    wcss.append(kmeans.inertia_)


# ------------------------------------------------------------
# 6. GRAFICO DELL'ELBOW METHOD
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

# Disegniamo la curva della WCSS al variare di K.
plt.plot(
    k_values,
    wcss,
    "o-",
    color="blue",
    markersize=8,
    label="WCSS"
)

# Evidenziamo il valore di K scelto osservando il gomito.
plt.axvline(
    x=3,
    color="red",
    linestyle="--",
    label="K candidato = 3"
)

plt.xlabel("Numero di cluster (K)")
plt.ylabel("WCSS (Inertia)")
plt.xticks(k_values)
plt.title("Elbow Method - Progetto 1")

plt.grid(True)

# La legenda deve essere aggiunta prima del salvataggio,
# altrimenti non comparirebbe nell'immagine esportata.
plt.legend()

# Miglioriamo la disposizione degli elementi del grafico.
plt.tight_layout()

# Salviamo il grafico in alta risoluzione.
plt.savefig(
    "progetto1_elbow.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 7. ADDESTRAMENTO DEL K-MEANS FINALE
# ------------------------------------------------------------

# In base al grafico Elbow, scegliamo K = 3.
kmeans_final = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=10,
    random_state=42
)

# Addestriamo il modello e assegniamo ogni punto
# al cluster corrispondente.
labels_final = kmeans_final.fit_predict(X_std)


# ------------------------------------------------------------
# 8. CALCOLO DELLA SILHOUETTE SCORE
# ------------------------------------------------------------

# La Silhouette Score valuta la qualità del clustering:
# considera quanto i punti sono vicini al proprio cluster
# e quanto sono separati dagli altri cluster.
#
# Utilizziamo X_std completo, cioè tutte e quattro
# le caratteristiche, perché il modello è stato
# addestrato utilizzando tutte e quattro.
silhouette = silhouette_score(
    X_std,
    labels_final
)

print("\nSilhouette Score:", silhouette)


# ------------------------------------------------------------
# 9. VISUALIZZAZIONE DEI CLUSTER E DEI CENTROIDI
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

# Disegniamo i punti del dataset.
# Il colore dipende dal cluster assegnato da K-Means.
#
# Per rappresentare i dati su un grafico bidimensionale
# utilizziamo soltanto le prime due caratteristiche:
# X_std[:, 0] = lunghezza del sepalo standardizzata
# X_std[:, 1] = larghezza del sepalo standardizzata
#
# Il modello, però, è stato addestrato su tutte e quattro.
plt.scatter(
    X_std[:, 0],
    X_std[:, 1],
    c=labels_final,
    cmap="viridis",
    s=40
)

# Recuperiamo le coordinate dei centroidi.
# Ogni centroide ha quattro coordinate perché K-Means
# è stato addestrato su quattro caratteristiche.
centroids = kmeans_final.cluster_centers_

# Disegniamo i centroidi sul grafico.
# Ne visualizziamo soltanto le prime due coordinate,
# per mantenerci sulle stesse due dimensioni dei punti.
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=200,
    color="red",
    edgecolors="black",
    label="Centroidi"
)

plt.title("K-Means Clustering - Iris (K=3)")
plt.xlabel("Lunghezza sepalo (standardizzata)")
plt.ylabel("Larghezza sepalo (standardizzata)")

plt.legend()
plt.grid(True)
plt.tight_layout()

# Salviamo il grafico finale in alta risoluzione.
plt.savefig(
    "progetto1_kmeans.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# FINE PROGETTO 1
# ============================================================
