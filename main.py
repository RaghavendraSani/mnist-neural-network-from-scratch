# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# CONFIGURATION
# file paths
TRAIN_PATH = "data/competition_train.csv"
TEST_PATH = "data/competition_test.csv"
SUBMISSION_PATH = "data/competition_submission.csv"

#random seed
RANDOM_SEED = 42

#dataset split
VALIDATION_SPLIT = 0.1

#visualisation
NUM_SAMPLE_IMAGES = 10
np.random.seed(RANDOM_SEED)


# NETWORK ARCHITECTURE
INPUT_SIZE = 784
HIDDEN1_SIZE = 512
HIDDEN2_SIZE = 512
OUTPUT_SIZE = 10


# DATA PREPROCESSING
def load_data():
    """
    Loads the training and testing datasets.
    Returns:
        train_df (DataFrame): Training dataset
        test_df (DataFrame): Testing dataset
    """
    print("Loading datasets...\n")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"training dataset shape: {train_df.shape}")
    print(f"testing dataset shape: {test_df.shape}")

    return train_df, test_df

def preprocess_data(train_df, test_df):
    """
    Separates features and labels from the datasets.
    Returns:
        X_train
        y_train
        X_test
    """
    #extract labels
    y_train = train_df["label"].to_numpy()
    #remove ImageId and label from training data
    X_train = train_df.drop(columns=['ImageId', 'label']).to_numpy()
    #remove ImageId from test
    X_test = test_df.drop(columns=['ImageId']).to_numpy()

    print("data preprocessing cmpleted.\n")
    print(f"training features shape: {X_train.shape}")
    print(f"training labels shape: {y_train.shape}")
    print(f"testing features shape: {X_test.shape}")

    assert X_train.shape[1] == 784, "Training data must have 784 features."
    assert X_test.shape[1] == 784, "Testing data must have 784 features."
    assert len(X_train) == len(y_train), \
        "Features and labels have different lengths."

    return X_train, y_train, X_test

def normalise_data(X_train, X_test):
    """
    Normalize pixel values from [0,255] to [0,1].
    """
    X_train = X_train.astype(np.float32)/255.0
    X_test = X_test.astype(np.float32)/255.0

    assert X_train.min() >= 0.0 and X_train.max() <= 1.0, \
        "Training pixels are not normalized."
    assert X_test.min() >= 0.0 and X_test.max() <= 1.0, \
        "Testing pixels are not normalized."

    print("\nNormalizing data...\n")
    print(f"training pixel range: {X_train.min():.3f}, {X_train.max():.3f}")
    print(f"testing pixel range: {X_test.min():.3f}, {X_test.max():.3f}\n")

    return X_train, X_test

def one_hot_encode(y, num_classes=10):
    """
    convert integer variables to one-hot encoded variables.
    """
    one_hot = np.zeros((len(y), num_classes), dtype=np.float32)
    one_hot[np.arange(len(y)), y] = 1

    assert one_hot.shape == (len(y), num_classes), \
        "Incorrect one-hot encoding shape."
    assert np.all(one_hot.sum(axis=1) == 1), \
        "Each one-hot vector must contain exactly one 1."

    print("One-Hot Encoding\n")
    print(f"Label Shape: {y.shape}")
    print(f"Encoded Shape: {one_hot.shape}\n")

    return one_hot

def train_validation_split(X, y_labels, y_one_hot, validation_split=VALIDATION_SPLIT):
    """
    Shuffle the dataset and split it into training and validation sets while keeping labels synchronized.
    """
    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)
    X = X[indices]
    y_labels = y_labels[indices]
    y_one_hot = y_one_hot[indices]
    split_index = int(num_samples * (1 - validation_split))
    X_train = X[:split_index]
    X_val = X[split_index:]
    y_train_labels = y_labels[:split_index]
    y_val_labels = y_labels[split_index:]
    y_train = y_one_hot[:split_index]
    y_val = y_one_hot[split_index:]

    assert X_train.shape[0] == y_train.shape[0], \
        "Training features and labels do not match."
    assert X_val.shape[0] == y_val.shape[0], \
        "Validation features and labels do not match."

    print("Train / Validation Split\n")
    print(f"Training Samples   : {len(X_train)}")
    print(f"Validation Samples : {len(X_val)}\n")

    return X_train, X_val,y_train, y_val, y_train_labels, y_val_labels

def visualize_samples(X, y, num_images=NUM_SAMPLE_IMAGES):
    """
    Display sample handwritten digits with their labels.
    """
    plt.figure(figsize=(12, 5))

    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap="gray")
        plt.title(f"Digit: {y[i]}", fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# NEURAL NETWORK
class NeuralNetwork:
    """
    Fully Connected Neural Network for MNIST Classification.
    """
    def __init__(self):
        """
        Initialize the neural network.
        """
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights using He Initialization.
        """
        self.W1 = np.random.randn(INPUT_SIZE, HIDDEN1_SIZE) * np.sqrt(2 / INPUT_SIZE)
        self.b1 = np.zeros((1,HIDDEN1_SIZE))

        self.W2 = np.random.randn(HIDDEN1_SIZE, HIDDEN2_SIZE ) * np.sqrt(2 / HIDDEN1_SIZE)
        self.b2 = np.zeros((1, HIDDEN2_SIZE))

        self.W3 = np.random.randn(HIDDEN2_SIZE, OUTPUT_SIZE) * np.sqrt(2 / HIDDEN2_SIZE)
        self.b3 = np.zeros((1, OUTPUT_SIZE))

        assert self.W1.shape == (INPUT_SIZE, HIDDEN1_SIZE)
        assert self.b1.shape == (1, HIDDEN1_SIZE)

        assert self.W2.shape == (HIDDEN1_SIZE, HIDDEN2_SIZE)
        assert self.b2.shape == (1, HIDDEN2_SIZE)

        assert self.W3.shape == (HIDDEN2_SIZE, OUTPUT_SIZE)
        assert self.b3.shape == (1, OUTPUT_SIZE)

        print("\nHe Weight Initialization Complete.\n")

        print(f"W1 Shape : {self.W1.shape}")
        print(f"b1 Shape : {self.b1.shape}")

        print(f"W2 Shape : {self.W2.shape}")
        print(f"b2 Shape : {self.b2.shape}")

        print(f"W3 Shape : {self.W3.shape}")
        print(f"b3 Shape : {self.b3.shape}")

    # ACTIVATION FUNCTIONS
    def relu(self, x):
        """
        Apply the ReLU activation function.
        """
        return np.maximum(0, x)

    def softmax(self, x):
        """
        Apply the Softmax activation function.
        """
        #numerical stability
        x = x-np.max(x, axis=1, keepdims=True)
        exp_values = np.exp(x)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

    # FORWARD PROPAGATION
    def forward(self, X):
        """
        Perform forward propagation through the network.
        """
        #Hidden layer 1
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)

        #Hidden layer 2
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.relu(self.Z2)

        #Output layer
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.softmax(self.Z3)

        assert self.Z1.shape == (X.shape[0], HIDDEN1_SIZE)
        assert self.Z2.shape == (X.shape[0], HIDDEN2_SIZE)
        assert self.Z3.shape == (X.shape[0], OUTPUT_SIZE)
        assert self.A1.shape == (X.shape[0], HIDDEN1_SIZE)
        assert self.A2.shape == (X.shape[0], HIDDEN2_SIZE)
        assert self.A3.shape == (X.shape[0], OUTPUT_SIZE)

        return self.A3

    # LOSS FUNCTION
    def compute_loss(self, y_true, y_pred):
        """
        Compute the loss of the network.
        """
        # prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

        assert np.isscalar(loss), \
            "Loss should be a scalar."
        assert loss >= 0, \
            "Loss should be non-negative."

        return loss




# MAIN
def main():
    train_df, test_df = load_data()
    X_train, y_labels, X_test = preprocess_data(train_df, test_df)
    X_train, X_test = normalise_data(X_train, X_test)
    y_one_hot = one_hot_encode(y_labels)
    X_train, X_val, y_train, y_val, y_train_labels, y_val_labels  = train_validation_split(X_train,y_labels,y_one_hot)
    visualize_samples(X_train, y_train_labels)

    model = NeuralNetwork()
    print("Neural Network object created successfully.\n")

    """"
    test_input = np.array([[2.0, 1.0, 0.1]])
    #print("Testing ReLU\n")
    #print("Input :", test_input)
    #print("Output:", model.relu(test_input))
    """
    """
    softmax_output = model.softmax(test_input)
    assert np.all(softmax_output >= 0), \
        "Softmax produced negative probabilities."
    assert np.all(softmax_output <= 1), \
        "Softmax probabilities exceed 1."
    assert np.allclose(np.sum(softmax_output, axis=1), 1.0), \
        "Softmax probabilities do not sum to 1."
    print("Testing Softmax")
    print("Input:\n", test_input)
    print("\nOutput:\n", softmax_output)
    print("\nRow Sum:", np.sum(softmax_output))
    """
    """
    print("Testing forward propagation..\n")
    test_batch = X_train[:5]
    output = model.forward(test_batch)
    print("Input shape: ", test_batch.shape)
    print("Output shape: ", output.shape)
    print("\nFirst prediction:\n")
    print(output[0])
    print("\nprobability sum:\n")
    print(np.sum(output[0]))
    """
    print("Testing loss function..\n")
    test_batch = X_train[:5]
    prediction = model.forward(test_batch)
    loss = model.compute_loss(y_train[:5], prediction)
    print("Prediction shape: ", prediction.shape)
    print(f"loss value: {loss:.6f}")

if __name__ == "__main__":
    main()