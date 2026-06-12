import numpy as np

from datasets import gaussians_dataset
from log_reg_students import LogisticRegression

# Set seed for reproducibility
np.random.seed(191090)


def main():
    """ Main execution script for training and evaluating Logistic Regression """

    # 1. Generate sample dataset using Gaussian distributions
    print("Generating dataset...")
    x_train, y_train, x_test, y_test = gaussians_dataset(2, [100, 150], [[1, 3], [-4, 8]], [[2, 3], [4, 1]])
    
    # 2. Instantiate the Logistic Regression model
    logistic_reg = LogisticRegression()
    
    # 3. Train the model using Gradient Descent
    print("Starting training...")
    logistic_reg.fit_gd(x_train, y_train, n_epochs=10000, learning_rate=0.01, verbose=True)
    
    # 4. Evaluate the model on the test set
    predictions = logistic_reg.predict(x_test)
    accuracy = float(np.sum(predictions == y_test)) / y_test.shape[0]
    
    print(f'\nTest accuracy: {accuracy:.4f}')
    
    # 5. Save model weights for future use
    print("Saving model weights to weights.npy...")
    np.save('weights.npy', logistic_reg._w)


if __name__ == '__main__':
    main()
