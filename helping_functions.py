import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import permutations

def plot_boxplots_for_dataframe_columns(data_frame):
    """
    Create boxplots for all columns in a pandas DataFrame to detect outliers.

    Args:
        data_frame (pd.DataFrame): The input DataFrame containing the data.

    Returns:
        None
    """
    plt.figure(figsize=(18, 3))  # Set the figure size

    for col in data_frame.columns:
        plt.subplot(1, len(data_frame.columns), list(data_frame.columns).index(col) + 1)
        plt.boxplot(data_frame[col], vert=False)
        plt.xlabel(col)

    plt.suptitle('Boxplots for DataFrame Columns', y=1.02)  # Add a title above the subplots
    plt.tight_layout()  # Ensure proper spacing between subplots
    plt.show()

# # Sample DataFrame
# data = {'Column1': [10, 15, 20, 25, 30, 35, 40, 45, 100],
#         'Column2': [5, 10, 15, 20, 25, 30, 35, 40, 45],
#         'Column3': [8, 12, 16, 24, 32, 48, 60, 72, 90]}

# df = pd.DataFrame(data)

# # Call the function with the sample DataFrame
# plot_boxplots_for_dataframe_columns(df)





def split_dataframe(data_frame, percentages):
    """
    Split a DataFrame into training, validation, and test DataFrames based on given percentages.

    Args:
        data_frame (pd.DataFrame): The input DataFrame.
        percentages (list): A list of three percentages [train_percent, valid_percent, test_percent].

    Returns:
        train_df (pd.DataFrame): Training DataFrame.
        valid_df (pd.DataFrame): Validation DataFrame.
        test_df (pd.DataFrame): Test DataFrame.
    """
    assert len(percentages) == 3, "Percentages list must contain three values."

    train_percent, valid_percent, test_percent = percentages
    assert train_percent + valid_percent + test_percent == 100, "Percentages must sum to 100."

    num_samples = len(data_frame)
    num_train = int(num_samples * train_percent / 100)
    num_valid = int(num_samples * valid_percent / 100)

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train_indices = indices[:num_train]
    valid_indices = indices[num_train:num_train + num_valid]
    test_indices = indices[num_train + num_valid:]

    train_df = data_frame.iloc[train_indices]
    valid_df = data_frame.iloc[valid_indices]
    test_df = data_frame.iloc[test_indices]

    print("Train DataFrame Shape:", train_df.shape)
    print("Validation DataFrame Shape:", valid_df.shape)
    print("Test DataFrame Shape:", test_df.shape)

    return train_df, valid_df, test_df

# # Sample DataFrame
# data = {'Column1': range(1, 11)}
# df = pd.DataFrame(data)

# # Percentages for splitting (60% train, 20% validation, 20% test)
# percentages = [60, 20, 20]

# # Split the DataFrame
# train_df, valid_df, test_df = split_dataframe(df, percentages)






def plot_training_history(history):
    """
    Plot training and validation accuracy and loss from a Keras training history.

    Args:
        history (dict): A Keras training history containing 'acc', 'val_acc', 'loss', and 'val_loss' keys.

    Returns:
        None
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # Plot training and validation accuracy
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

    # Plot training and validation loss
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

# Example usage:
# history = {
#     'acc': [0.6, 0.7, 0.8, 0.9],
#     'val_acc': [0.5, 0.6, 0.7, 0.8],
#     'loss': [0.5, 0.4, 0.3, 0.2],
#     'val_loss': [0.6, 0.5, 0.4, 0.3]
# }



# from itertools import permutations

def generate_permutation_dict(arr):
    """
    Generate a dictionary of permutations for an input array.

    Parameters:
    arr (numpy.ndarray or list): Input array containing values to be permuted.

    Returns:
    dict: A dictionary where keys are permutation labels and values are arrays
          representing the input array with values permuted according to each
          permutation.
    """
    # Defining the replacement values and their corresponding permutations
    replacements = [0, 1, 2]
    permutations_list = list(permutations(replacements))
    
    # Initializing a dictionary to store the reversed permutations
    reversed_permutations_dict = {}
    
    for i, perm in enumerate(permutations_list):
        # Creating a key that shows the current permutation
        key = f'Permutation_{i+1}: {perm}'
        
        # Replacing values in the input array using the current permutation
        replaced_arr = np.array([perm[replacements.index(val)] for val in arr])
        
        # Storing the resulting array in the dictionary
        reversed_permutations_dict[key] = replaced_arr
    
    return reversed_permutations_dict

# input_arr = [0, 1, 2]
# result    = generate_permutation_dict(input_arr)
# print(result)




#import numpy as np

def optimize_weights_with_random_search(true_labels, model_predictions, num_iterations=10000):
    """
    Optimize model weights for an ensemble using random search.

    Args:
        true_labels (numpy.ndarray): True labels for the data samples.
        model_predictions (list of numpy.ndarray): List of model predictions,
            where each prediction array has shape (number_of_samples, number_of_classes).
        num_iterations (int, optional): Number of random search iterations. Default is 10,000.

    Returns:
        tuple: A tuple containing the best weights for the models and the corresponding accuracy.

    Example:
        # Replace the following with your true_labels and model_predictions
        true_labels       = valid_labels
        model_predictions = [gmm_valid_probs, NN_valid_probs, XGB_valid_probs, RF_valid_probs]

        best_weights, best_accuracy = optimize_weights_with_random_search(true_labels, model_predictions)

    """
    # Initializing weights for each model
    initial_weights = np.ones(len(model_predictions)) / len(model_predictions)

    # Define a function to compute the weighted average of predictions
    def weighted_average_predictions(weights):
        ensemble_predictions = np.zeros_like(model_predictions[0])
        for i, model_pred in enumerate(model_predictions):
            ensemble_predictions += weights[i] * model_pred
        return ensemble_predictions

    # Define a function to calculate the accuracy of predictions
    def calculate_accuracy(predictions):
        predicted_labels = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)
        return accuracy

    # Define a function to perform random search optimization
    def random_search_optimization():
        best_accuracy = 0.0
        best_weights = None

        for _ in range(num_iterations):
            # Generate random weights for each model
            random_weights = np.random.rand(len(model_predictions))
            random_weights /= np.sum(random_weights)  # Normalize to ensure weights sum to 1

            # Calculate ensemble predictions
            ensemble_predictions = weighted_average_predictions(random_weights)

            # Calculate accuracy with the random weights
            accuracy = calculate_accuracy(ensemble_predictions)

            # Update best weights if accuracy is improved
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = random_weights

        return best_weights, best_accuracy

    # Performing random search optimization
    best_weights, best_accuracy = random_search_optimization()

    return best_weights, best_accuracy





