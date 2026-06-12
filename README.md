# artificial-intelligence
Some exercises to learn how artificial intelligence works.

## Models Description (English)

### Naive Bayes
- **Prior Calculation**: computes class prior probability based on its frequency within the training set.
- **Statistical Parameters**: calculates mean and variance for each feature to describe the class distribution.
- **Log-Likelihood**: uses the Gaussian probability density function to estimate the likelihood of data belonging to a category.
- **Prediction**: selects the class that maximizes the posterior probability by combining priors and likelihoods.
```text
# Training
For each class c:
    P(c) = count(c) / N
    μ_c = mean(X_c)
    σ²_c = var(X_c)

# Prediction
P(x|c) = GaussianPDF(x, μ_c, σ²_c)
Posterior(c) = log(P(c)) + Σ log(P(x_i|c))
Result = argmax(Posterior)
```

### Linear Discriminant Analysis (LDA)
- **Means and Priors**: calculates the feature mean vector and the prior probability for each class.
- **Common Covariance**: derives a shared scatter matrix that models the variance of the entire dataset.
- **Mahalanobis Distance**: evaluates sample similarity to classes using the inverse of the shared covariance matrix.
- **Prediction**: assigns the label of the class that maximizes the calculated discriminant function.
```text
# Parameters
μ_c = mean(X_c)
P(c) = count(c) / N
Σ = (1/(N-K)) * Σ_c Σ_{x in c} (x - μ_c)(x - μ_c)ᵀ

# Prediction
δ_c(x) = xᵀ Σ⁻¹ μ_c - 0.5 * μ_cᵀ Σ⁻¹ μ_c + log(P(c))
Result = argmax(δ_c(x))
```

### Logistic Regression
- **Sigmoid Function**: maps the output of a linear combination to a probability value between 0 and 1.
- **Gradient Descent**: iteratively updates model weights to minimize the classification error (loss).
- **Loss Derivative**: computes the gradient direction to adjust weights based on the difference between predictions and targets.
- **Prediction**: assigns a binary class by applying a threshold (usually 0.5) to the sigmoid output.
```text
# Training (Gradient Descent)
σ(z) = 1 / (1 + exp(-z))
gradient = (1/N) * Xᵀ * (σ(Xw) - y)
w = w - η * gradient

# Prediction
p = σ(X_test * w)
Result = 1 if p >= 0.5 else 0
```

### Support Vector Machines (SVM)
- **Target Mapping**: converts original labels into -1 and 1 to fit the mathematical margin formalism.
- **Margin Optimization**: searches for the hyperplane that maximizes the distance between points of different classes.
- **Hinge Loss**: applies a cost function that penalizes data points that violate the separation margin.
- **Prediction**: classifies new data points based on the sign of the product between weights and features.
```text
# Optimization
y ∈ {-1, 1}
Minimize L = (1/2)||w||² + C * Σ max(0, 1 - y_i(wᵀx_i + b))

# Prediction
Result = sign(wᵀx + b)
```

### Adaboost
- **Sample Weights**: initializes uniform importance for all data, adjusting it over time to focus on errors.
- **Weak Classifier**: trains a sequence of simple models on weighted subsets of the original dataset.
- **Alpha Calculation**: assigns a weight to each weak model based on its specific ability to reduce error.
- **Weight Update**: increases the importance of misclassified examples to force the next learner to correct them.
- **Final Combination**: produces the final prediction through a weighted vote of all weak classifiers.
```text
D_1(i) = 1/N
For t = 1 to T:
    Train weak classifier h_t minimizing error ε_t
    α_t = 0.5 * ln((1 - ε_t) / ε_t)
    D_{t+1}(i) = D_t(i) * exp(-α_t * y_i * h_t(x_i)) / Z_t

Final H(x) = sign(Σ α_t * h_t(x))
```

### K-Means
- **Centers Initialization**: randomly selects k points in the feature space to represent initial cluster centers.
- **Cluster Assignment**: associates each data point to the nearest centroid using Euclidean distance.
- **Means Update**: recalculates centroid positions as the average value of all points assigned to that cluster.
- **Cost Optimization**: repeats the process to minimize the sum of squared internal distances (inertia).
```text
Initialize centroids μ_1...μ_k
Repeat:
    Assign: c(i) = argmin_j ||x_i - μ_j||²
    Update: μ_j = mean(x_i where c(i) == j)
    Minimize: J = Σ ||x_i - μ_{c(i)}||²
```

### Spectral Clustering
- **Affinity Matrix**: builds a similarity graph between points using a Gaussian (RBF) kernel.
- **Normalized Laplacian**: creates a matrix capturing the geometric structure and connectivity of the dataset.
- **Spectral Decomposition**: calculates eigenvalues and eigenvectors to project data into a low-dimensional space.
- **Fiedler Vector**: uses the second smallest eigenvector to partition the data into coherent groups.
```text
A_ij = exp(-||x_i - x_j||² / (2σ²))
L = D - A
Compute eigenvectors of L
Project data using first k eigenvectors
Cluster = KMeans(ProjectedData, clusters=k)
```

---

## Descrizione dei Modelli

### Naive Bayes
- **Calcolo dei Prior**: si ricava la probabilità a priori di ogni classe basandosi sulla sua frequenza nel dataset.
- **Parametri Statistici**: si calcolano media e varianza di ogni feature per descrivere la distribuzione di ogni classe.
- **Log-Likelihood**: si usa la densità di probabilità Gaussiana per stimare quanto i dati appartengano a una categoria.
- **Predizione**: si seleziona la classe con la massima probabilità posteriore combinando prior e likelihood.
```text
# Addestramento
Per ogni classe c:
    P(c) = conteggio(c) / N
    μ_c = media(X_c)
    σ²_c = varianza(X_c)

# Predizione
P(x|c) = PDF_Gaussiana(x, μ_c, σ²_c)
Posteriori(c) = log(P(c)) + Σ log(P(x_i|c))
Risultato = argmax(Posteriori)
```

### Linear Discriminant Analysis (LDA)
- **Medie e Prior**: si calcolano il vettore medio delle caratteristiche e la probabilità a priori per ogni classe.
- **Covarianza Comune**: si ricava una matrice di dispersione condivisa che modella la varianza dell'intero dataset.
- **Distanza di Mahalanobis**: si valuta la somiglianza dei campioni alle classi tramite la matrice di covarianza inversa.
- **Predizione**: il modello assegna l'etichetta della classe che massimizza la funzione discriminante calcolata.
```text
# Parametri
μ_c = media(X_c)
P(c) = conteggio(c) / N
Σ = (1/(N-K)) * Σ_c Σ_{x in c} (x - μ_c)(x - μ_c)ᵀ

# Predizione
δ_c(x) = xᵀ Σ⁻¹ μ_c - 0.5 * μ_cᵀ Σ⁻¹ μ_c + log(P(c))
Risultato = argmax(δ_c(x))
```

### Logistic Regression
- **Funzione Sigmoide**: trasforma il risultato di una combinazione lineare in una probabilità tra 0 e 1.
- **Gradient Descent**: aggiorna iterativamente i pesi del modello per minimizzare l'errore di classificazione.
- **Derivata della Loss**: calcola la direzione del gradiente per correggere i pesi in base alla differenza tra predizione e target.
- **Predizione**: assegna la classe binaria applicando una soglia (solitamente 0.5) al valore della sigmoide.
```text
# Addestramento (Gradient Descent)
σ(z) = 1 / (1 + exp(-z))
gradiente = (1/N) * Xᵀ * (σ(Xw) - y)
w = w - η * gradiente

# Predizione
p = σ(X_test * w)
Risultato = 1 se p >= 0.5 altrimenti 0
```

### Support Vector Machines (SVM)
- **Mappatura Target**: converte le etichette originali in valori -1 e 1 per adattarle al formalismo del margine.
- **Ottimizzazione del Margine**: cerca l'iperpiano che massimizza la distanza tra i punti delle diverse classi.
- **Hinge Loss**: applica una funzione di costo che penalizza i punti che violano il margine di separazione.
- **Predizione**: classifica i nuovi dati in base al segno del prodotto tra il vettore dei pesi e le caratteristiche.
```text
# Ottimizzazione
y ∈ {-1, 1}
Minimizza L = (1/2)||w||² + C * Σ max(0, 1 - y_i(wᵀx_i + b))

# Predizione
Risultato = segno(wᵀx + b)
```

### Adaboost
- **Pesi dei Campioni**: inizializza un'importanza uniforme per tutti i dati, che varia per focalizzarsi sugli errori.
- **Weak Classifier**: addestra una sequenza di modelli semplici su sottoinsiemi pesati del dataset originale.
- **Calcolo di Alpha**: assegna un peso a ogni modello debole in base alla sua capacità di ridurre l'errore.
- **Aggiornamento Pesi**: aumenta l'importanza degli esempi classificati male per forzare il prossimo learner a correggerli.
- **Combinazione Finale**: produce la predizione finale tramite una votazione pesata di tutti i classificatori deboli.
```text
D_1(i) = 1/N
Per t = 1 a T:
    Addestra weak classifier h_t minimizzando l'errore ε_t
    α_t = 0.5 * ln((1 - ε_t) / ε_t)
    D_{t+1}(i) = D_t(i) * exp(-α_t * y_i * h_t(x_i)) / Z_t

H(x) finale = segno(Σ α_t * h_t(x))
```

### K-Means
- **Inizializzazione Centri**: sceglie casualmente k punti nello spazio per rappresentare i centri iniziali dei cluster.
- **Assegnazione Cluster**: associa ogni punto del dataset al centroide più vicino tramite distanza euclidea.
- **Aggiornamento Medie**: ricalcola la posizione dei centroidi come valore medio di tutti i punti assegnati al cluster.
- **Ottimizzazione del Costo**: ripete il processo per minimizzare la somma dei quadrati delle distanze interne.
```text
Inizializza centroidi μ_1...μ_k
Ripeti:
    Assegnazione: c(i) = argmin_j ||x_i - μ_j||²
    Aggiornamento: μ_j = media(x_i dove c(i) == j)
    Minimizza: J = Σ ||x_i - μ_{c(i)}||²
```

### Spectral Clustering
- **Matrice di Affinità**: costruisce un grafo di similarità tra punti utilizzando un kernel Gaussiano (RBF).
- **Laplaciana Normalizzata**: crea una matrice che cattura la struttura geometrica del dataset e le sue connessioni.
- **Decomposizione Spettrale**: calcola autovalori e autovettori per proiettare i dati in uno spazio a bassa dimensionalità.
- **Fiedler Vector**: utilizza il secondo autovettore più piccolo per separare i dati in gruppi coerenti.
```text
A_ij = exp(-||x_i - x_j||² / (2σ²))
L = D - A
Calcola autovettori di L
Proietta i dati usando i primi k autovettori
Cluster = KMeans(DatiProiettati, clusters=k)
```
