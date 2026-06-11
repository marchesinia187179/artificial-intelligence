# artificial-intelligence
Some exercises to learn how artificial intelligence works.

## Models Description (English)

### Naive Bayes
- **Prior Calculation**: computes class prior probability based on its frequency within the training set.
- **Statistical Parameters**: calculates mean and variance for each feature to describe the class distribution.
- **Log-Likelihood**: uses the Gaussian probability density function to estimate the likelihood of data belonging to a category.
- **Prediction**: selects the class that maximizes the posterior probability by combining priors and likelihoods.

### Linear Discriminant Analysis (LDA)
- **Means and Priors**: calculates the feature mean vector and the prior probability for each class.
- **Common Covariance**: derives a shared scatter matrix that models the variance of the entire dataset.
- **Mahalanobis Distance**: evaluates sample similarity to classes using the inverse of the shared covariance matrix.
- **Prediction**: assigns the label of the class that maximizes the calculated discriminant function.

### Logistic Regression
- **Sigmoid Function**: maps the output of a linear combination to a probability value between 0 and 1.
- **Gradient Descent**: iteratively updates model weights to minimize the classification error (loss).
- **Loss Derivative**: computes the gradient direction to adjust weights based on the difference between predictions and targets.
- **Prediction**: assigns a binary class by applying a threshold (usually 0.5) to the sigmoid output.

### Support Vector Machines (SVM)
- **Target Mapping**: converts original labels into -1 and 1 to fit the mathematical margin formalism.
- **Margin Optimization**: searches for the hyperplane that maximizes the distance between points of different classes.
- **Hinge Loss**: applies a cost function that penalizes data points that violate the separation margin.
- **Prediction**: classifies new data points based on the sign of the product between weights and features.

### Adaboost
- **Sample Weights**: initializes uniform importance for all data, adjusting it over time to focus on errors.
- **Weak Classifier**: trains a sequence of simple models on weighted subsets of the original dataset.
- **Alpha Calculation**: assigns a weight to each weak model based on its specific ability to reduce error.
- **Weight Update**: increases the importance of misclassified examples to force the next learner to correct them.
- **Final Combination**: produces the final prediction through a weighted vote of all weak classifiers.

### K-Means
- **Centers Initialization**: randomly selects k points in the feature space to represent initial cluster centers.
- **Cluster Assignment**: associates each data point to the nearest centroid using Euclidean distance.
- **Means Update**: recalculates centroid positions as the average value of all points assigned to that cluster.
- **Cost Optimization**: repeats the process to minimize the sum of squared internal distances (inertia).

### Spectral Clustering
- **Affinity Matrix**: builds a similarity graph between points using a Gaussian (RBF) kernel.
- **Normalized Laplacian**: creates a matrix capturing the geometric structure and connectivity of the dataset.
- **Spectral Decomposition**: calculates eigenvalues and eigenvectors to project data into a low-dimensional space.
- **Fiedler Vector**: uses the second smallest eigenvector to partition the data into coherent groups.

---

## Descrizione dei Modelli

### Naive Bayes
- **Calcolo dei Prior**: si ricava la probabilità a priori di ogni classe basandosi sulla sua frequenza nel dataset.
- **Parametri Statistici**: si calcolano media e varianza di ogni feature per descrivere la distribuzione di ogni classe.
- **Log-Likelihood**: si usa la densità di probabilità Gaussiana per stimare quanto i dati appartengano a una categoria.
- **Predizione**: si seleziona la classe con la massima probabilità posteriore combinando prior e likelihood.

### Linear Discriminant Analysis (LDA)
- **Medie e Prior**: si calcolano il vettore medio delle caratteristiche e la probabilità a priori per ogni classe.
- **Covarianza Comune**: si ricava una matrice di dispersione condivisa che modella la varianza dell'intero dataset.
- **Distanza di Mahalanobis**: si valuta la somiglianza dei campioni alle classi tramite la matrice di covarianza inversa.
- **Predizione**: il modello assegna l'etichetta della classe che massimizza la funzione discriminante calcolata.

### Logistic Regression
- **Funzione Sigmoide**: trasforma il risultato di una combinazione lineare in una probabilità tra 0 e 1.
- **Gradient Descent**: aggiorna iterativamente i pesi del modello per minimizzare l'errore di classificazione.
- **Derivata della Loss**: calcola la direzione del gradiente per correggere i pesi in base alla differenza tra predizione e target.
- **Predizione**: assegna la classe binaria applicando una soglia (solitamente 0.5) al valore della sigmoide.

### Support Vector Machines (SVM)
- **Mappatura Target**: converte le etichette originali in valori -1 e 1 per adattarle al formalismo del margine.
- **Ottimizzazione del Margine**: cerca l'iperpiano che massimizza la distanza tra i punti delle diverse classi.
- **Hinge Loss**: applica una funzione di costo che penalizza i punti che violano il margine di separazione.
- **Predizione**: classifica i nuovi dati in base al segno del prodotto tra il vettore dei pesi e le caratteristiche.

### Adaboost
- **Pesi dei Campioni**: inizializza un'importanza uniforme per tutti i dati, che varia per focalizzarsi sugli errori.
- **Weak Classifier**: addestra una sequenza di modelli semplici su sottoinsiemi pesati del dataset originale.
- **Calcolo di Alpha**: assegna un peso a ogni modello debole in base alla sua capacità di ridurre l'errore.
- **Aggiornamento Pesi**: aumenta l'importanza degli esempi classificati male per forzare il prossimo learner a correggerli.
- **Combinazione Finale**: produce la predizione finale tramite una votazione pesata di tutti i classificatori deboli.

### K-Means
- **Inizializzazione Centri**: sceglie casualmente k punti nello spazio per rappresentare i centri iniziali dei cluster.
- **Assegnazione Cluster**: associa ogni punto del dataset al centroide più vicino tramite distanza euclidea.
- **Aggiornamento Medie**: ricalcola la posizione dei centroidi come valore medio di tutti i punti assegnati al cluster.
- **Ottimizzazione del Costo**: ripete il processo per minimizzare la somma dei quadrati delle distanze interne.

### Spectral Clustering
- **Matrice di Affinità**: costruisce un grafo di similarità tra punti utilizzando un kernel Gaussiano (RBF).
- **Laplaciana Normalizzata**: crea una matrice che cattura la struttura geometrica del dataset e le sue connessioni.
- **Decomposizione Spettrale**: calcola autovalori e autovettori per proiettare i dati in uno spazio a bassa dimensionalità.
- **Fiedler Vector**: utilizza il secondo autovettore più piccolo per separare i dati in gruppi coerenti.
