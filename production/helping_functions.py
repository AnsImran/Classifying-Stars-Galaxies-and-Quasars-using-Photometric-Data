import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import permutations
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pickle
from tensorflow.keras.models import load_model






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

    # Plotting training and validation accuracy
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

    # Plotting training and validation loss
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
    Generate a dictionary of shuffled arrays using the input array & permutations of [0, 1, 2].
    The array is shuffled based on the different permutations of [0, 1, 2]

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

# input_arr = [0, 1, 2, 0, 1, 2, 2, 1, 0]
# result    = generate_permutation_dict(input_arr)
# print(result)





# import numpy as np
# from itertools import permutations

def shuffle_columns(input_indices, input_array):
    """
    Shuffle columns of the input array based on permutations of input_indices.

    Parameters:
    input_indices (list): List of indices representing column order.
    input_array (numpy.ndarray): Input array to be shuffled.

    Returns:
    dict: A dictionary where keys are formatted as "permutation(0, 1, 2)" and values are shuffled arrays.
    """
    permutation_dict = {}

    # Generate all permutations of input_indices
    index_permutations = permutations(input_indices)

    for perm in index_permutations:
        # Create a key based on the current permutation
        key = f"permutation({', '.join(map(str, perm))})"

        # Shuffle columns of input_array based on the current permutation
        shuffled_array = input_array[:, list(perm)]

        # Store the shuffled array in the dictionary
        permutation_dict[key] = shuffled_array

    return permutation_dict

# # Example usage:
# input_indices = [0, 1, 2]
# input_array = np.array([[1, 2, 3],
#                         [4, 5, 6],
#                         [7, 8, 9]])

# result_dict = shuffle_columns(input_indices, input_array)

# # Print the shuffled arrays for each permutation with formatted keys
# for key, shuffled_array in result_dict.items():
#     print(f"{key}:")
#     print(shuffled_array)
#     print()








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







def score_calculator(probs_array, true_labels, shape_array):
    score = np.sum(np.argmax(probs_array, axis=1) == true_labels) / shape_array.shape[0]
    return score

# e.g:
# score_calculator(gmm_test_probs, test_labels, test_np)


def score_calculator_for_preds(preds_array, true_labels, shape_array):
    score = np.sum(preds_array == true_labels) / shape_array.shape[0]
    return score

# e.g:
# score_calculator_for_preds(gmm_valid_preds, valid_labels, valid_np)










#from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def classification_report_and_confusion_matrix(true_labels, predicted_labels):
    """
    Calculate and display a classification report and a confusion matrix.

    Parameters:
    true_labels (array-like): Ground truth (correct) target labels.
    predicted_labels (array-like): Predicted target labels.

    Returns:
    None: The function prints the classification report and displays the confusion matrix.
    """
    # Generating the classification report
    report = classification_report(true_labels, predicted_labels)
    print(report)
    
    # Generating and displaying the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    disp        = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                         display_labels=['Quasar', 'Galaxy', 'Star'])
    disp.plot(cmap='Blues')













# import pandas as pd
# import numpy as np

def probabilities_to_dataframe(probabilities_array):
    """
    Convert a 2D NumPy array of probabilities into a well-structured DataFrame.
    It can only handle 3 classes i.e in your probabilities_array, there are probabilities corresponding to only 3 classes

    Args:
    probabilities_array (numpy.ndarray): A 2D array with rows representing different samples
        and columns representing class probabilities.

    Returns:
    pandas.DataFrame: A DataFrame with columns for class probabilities, predicted labels, and
        the maximum normalized probability for each sample.

    Example:
    >>> probabilities_array = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3], [0.4, 0.5, 0.1]])
    >>> result_df = probabilities_to_dataframe(probabilities_array)
    >>> print(result_df)
    """

    # Creating a DataFrame from the probabilities_array
    df = pd.DataFrame(probabilities_array, columns=['prob_of_being_Quasar', 'prob_of_being_Galaxy', 'prob_of_being_Star'])

    # Normalizing the probabilities in each row
    df = df.div(df.sum(axis=1), axis=0)

    # Adding a 'Predicted_Label' column with the index of the class with the highest probability
    df['predicted_label'] = df.idxmax(axis=1)

    # Defining a dictionary to specify the replacements
    label_replacements = {'prob_of_being_Quasar': 'quasar', 'prob_of_being_Galaxy': 'galaxy', 'prob_of_being_Star': 'star'}

    # Using the map method to perform the replacements
    df['predicted_label'] = df['predicted_label'].map(label_replacements)

    # Adding a 'Normalized_Probability_of_Predicted_Class' column with the value of the highest predicted probability
    df['Probability_of_Predicted_Class'] = df[['prob_of_being_Quasar', 'prob_of_being_Galaxy', 'prob_of_being_Star']].max(axis=1)

    df.index.name = 'sample_no.'

    return df

# Example usage:
# Assuming probabilities_array is your array of predictions
# Replace this with your actual predictions
# probabilities_array = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3], [0.4, 0.5, 0.1]])
# result_df = probabilities_to_dataframe(probabilities_array)
# result_df













# import pandas as pd
# import numpy as np

def classification_probability_displayer(Probabilities, True_Labels):
    """
    Display classification statistics and probabilities based on input probabilities and true labels.

    Args:
        Probabilities (list or array): Predicted class probabilities for each sample.
        True_Labels (list or array): True class labels for each sample.

    Returns:
        None
    """
    # Convert probabilities to a DataFrame
    data_frame_for_probabilities_of_current_model = probabilities_to_dataframe(Probabilities)
    
    # Add true class labels to the DataFrame
    data_frame_for_probabilities_of_current_model['True_class'] = True_Labels
    
    # Define a dictionary to specify the replacements for class labels
    label_replacements = {0: 'quasar', 1: 'galaxy', 2: 'star'}
    
    # Use the map method to perform the replacements for true class labels
    data_frame_for_probabilities_of_current_model['True_class'] = data_frame_for_probabilities_of_current_model['True_class'].map(label_replacements)
    
    # Group the DataFrame by predicted labels
    groups_df = data_frame_for_probabilities_of_current_model.groupby('predicted_label')       ###################################
    
    # Calculate total number of samples
    total_no_of_samples = len(True_Labels)
    print(f'total_no_of_samples: {total_no_of_samples}')
    
    # Calculate the number of samples of each class
    no_of_samples_of_each_class = len(True_Labels) / len(groups_df.groups)
    print(f'no_of_samples_of_each_class: {no_of_samples_of_each_class}\n')
    
    # Calculate and display statistics for Quasars
    predicted_quasars_df = groups_df.get_group('quasar')                                       #####################################
    predicted_percentage_of_quasars = (len(predicted_quasars_df) / total_no_of_samples) * 100
    print(f'predicted_percentage_of_quasars: {predicted_percentage_of_quasars}')
    true_percentage_of_quasars = (no_of_samples_of_each_class / total_no_of_samples) * 100
    print(f'true_percentage_of_quasars: {true_percentage_of_quasars}\n')
    
    # Calculate and display statistics for Galaxies
    predicted_galaxies_df = groups_df.get_group('galaxy')
    predicted_percentage_of_galaxies = (len(predicted_galaxies_df) / total_no_of_samples) * 100
    print(f'predicted_percentage_of_galaxies: {predicted_percentage_of_galaxies}')
    true_percentage_of_galaxies = (no_of_samples_of_each_class / total_no_of_samples) * 100
    print(f'true_percentage_of_galaxies: {true_percentage_of_galaxies}\n')
    
    # Calculate and display statistics for Stars
    predicted_stars_df = groups_df.get_group('star')
    predicted_percentage_of_stars = (len(predicted_stars_df) / total_no_of_samples) * 100
    print(f'predicted_percentage_of_stars: {predicted_percentage_of_stars}')
    true_percentage_of_stars = (no_of_samples_of_each_class / total_no_of_samples) * 100
    print(f'true_percentage_of_stars: {true_percentage_of_stars}\n')
    
    # Function to calculate statistics for misclassified and correctly classified samples
    def calculate_statistics(class_df, label):
        print(f'Total no. of samples classified as {label}: {len(class_df)}')                                        ####################################
        mask = class_df['predicted_label'] != class_df['True_class']
        indices = groups_df.get_group(label).index[mask].tolist()
        no_of_samples_misclassified = len(indices)
        print(f'no_of_samples_confused_as_{label}s: {no_of_samples_misclassified}')
    
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
            samples_misclassified_with_probability = (np.sum(class_df.loc[indices]['Probability_of_Predicted_Class'] > threshold) / no_of_samples_misclassified) * 100
            print(f'samples_confused_as_{label}s_with_probability_greater_than_{threshold}: {samples_misclassified_with_probability} %')
    
        mask = class_df['predicted_label'] == class_df['True_class']
        indices = groups_df.get_group(label).index[mask].tolist()
        no_of_samples_correctly_classified = len(indices); print()
        print(f'no_of_samples_correctly_classified_as_{label}s: {no_of_samples_correctly_classified}')
    
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
            samples_correctly_classified_with_probability = (np.sum(class_df.loc[indices]['Probability_of_Predicted_Class'] > threshold) / no_of_samples_correctly_classified) * 100
            print(f'samples_correctly_classified_as_{label}s_with_probability_greater_than_{threshold}: {samples_correctly_classified_with_probability} %')
        print('\n\n')
    
    # Calculate and display statistics for misclassified and correctly classified Quasars
    calculate_statistics(predicted_quasars_df, 'quasar')
    
    # Calculate and display statistics for misclassified and correctly classified Galaxies
    calculate_statistics(predicted_galaxies_df, 'galaxy')
    
    # Calculate and display statistics for misclassified and correctly classified Stars
    calculate_statistics(predicted_stars_df, 'star')









def predictions_corresponding_to_correct_permutation(data_points_np_normalized, true_labels, gmm_model):
    """
    Obtain GMM predicted probabilities for the correct permutation of features.

    Args:
        data_points_np_normalized (array): Normalized data points.
        true_labels (array): True class labels.
        gmm_model: Trained Gaussian Mixture Model.

    Returns:
        array: GMM predicted probabilities for the correct permutation of features.
    """
    # Get GMM predicted probabilities for the input data
    gmm_probs = gmm_model.predict_proba(data_points_np_normalized)
    
    # Define the input indices for column permutation
    input_indices = [0, 1, 2]
    
    # Generate a dictionary with probabilities for all permutations
    dict_probs = shuffle_columns(input_indices, gmm_probs)
    
    # Initialize variables to track the best permutation
    highest_score = 0
    max_key = 0
    
    # Find the permutation with the highest score
    for key, value in dict_probs.items():
        current_score = score_calculator(value, true_labels, data_points_np_normalized)
        
        if current_score > highest_score:
            highest_score = current_score
            max_key = key

    # Update GMM probabilities with the best permutation
    gmm_probs = dict_probs[max_key]
    
    return gmm_probs

# gmm_probs = predictions_corresponding_to_correct_permutation(data_points_np_normalized, true_labels)











# import pickle
# from tensorflow.keras.models import load_model
# import numpy as np

def ensemble_of_models_with_true_labels(input_data_normalized, true_labels, best_weights):
    """
    Create an ensemble of machine learning models, calculate probabilities, and evaluate the performance.

    This function combines the predictions of Gaussian Mixture Model (GMM), Neural Network (NN),
    XGBoost (XGB), and Random Forest (RF) models using specified weights and evaluates the
    performance based on true labels.

    Args:
    input_data_normalized (numpy.ndarray): Normalized input data.
    true_labels (numpy.ndarray): True labels for the input data.
    best_weights (list): List of weights for each model in the order [GMM, NN, XGB, RF].

    Returns:
    None

    Example:
    >>> best_weights = [0.2, 0.3, 0.2, 0.3]
    >>> input_data_normalized = load_input_data('input_data_normalized.pkl')
    >>> true_labels = load_labels('true_labels.pkl')
    >>> ensemble_of_models_with_true_labels(input_data_normalized, true_labels, best_weights)
    """

    # Loading the saved GMM model from the file
    with open('gmm_model.pkl', 'rb') as file:
        gmm_model = pickle.load(file)
    gmm_probs = gmm_model.predict_proba(input_data_normalized)

    # Shuffle GMM model probabilities
    input_indices = [0, 1, 2]
    probs_dict = shuffle_columns(input_indices, gmm_probs)
    gmm_probs = probs_dict['permutation(1, 0, 2)']

    # Loading the saved NN model
    NN_model = load_model('NN_model.h5')
    NN_probs = NN_model.predict(input_data_normalized)

    # Loading the saved XGBoost model from the file
    with open('XGB_model.pkl', 'rb') as file:
        XGB_model = pickle.load(file)
    XGB_probs = XGB_model.predict_proba(input_data_normalized)

    # Loading the saved RF model from the file
    with open('RF_model.pkl', 'rb') as file:
        RF_model = pickle.load(file)
    RF_probs = RF_model.predict_proba(input_data_normalized)

    # Collecting model predictions for validation data
    model_predictions = [gmm_probs, NN_probs, XGB_probs, RF_probs]

    # Assigning the best weights obtained from random search
    gmm_weight = best_weights[0]
    NN_weight = best_weights[1]
    XGB_weight = best_weights[2]
    RF_weight = best_weights[3]

    # Calculating final probabilities using the best weights for each model
    final_probs = (gmm_weight * gmm_probs + NN_weight * NN_probs +
                   XGB_weight * XGB_probs + RF_weight * RF_probs) / 4

    # Converting probabilities into predictions
    final_predicted_labels = np.argmax(final_probs, axis=1)

    # Generating a classification report and displaying a confusion matrix
    classification_report_and_confusion_matrix(true_labels, final_predicted_labels)

    # Calculating accuracy for the final predictions
    score = score_calculator_for_preds(final_predicted_labels, true_labels, input_data_normalized)
    print(f'Accuracy: {score}')

    classification_probability_displayer(final_probs, true_labels)










# import pickle
# from tensorflow.keras.models import load_model
# import pandas as pd
# import numpy as np

def ensemble_of_models_without_true_labels(input_data_normalized, best_weights):
    """
    Create an ensemble of machine learning models and calculate probabilities for each class.

    This function combines the predictions of Gaussian Mixture Model (GMM), Neural Network (NN),
    XGBoost (XGB), and Random Forest (RF) models using specified weights and returns the
    final probabilities.

    Args:
    input_data_normalized (numpy.ndarray): Normalized input data.
    best_weights (list): List of weights for each model in the order [GMM, NN, XGB, RF].

    Returns:
    pandas.DataFrame: A DataFrame with columns for class probabilities, predicted labels, and
        the maximum normalized probability for each sample.

    Example:
    >>> best_weights = [0.2, 0.3, 0.2, 0.3]
    >>> probabilities_array = ensemble_of_models_without_true_labels(input_data_normalized, best_weights)
    >>> print(probabilities_array)
    """

    # Loading the saved GMM model from the file
    with open('gmm_model.pkl', 'rb') as file:
        gmm_model = pickle.load(file)
    gmm_probs = gmm_model.predict_proba(input_data_normalized)

    # Loading the saved NN model
    NN_model = load_model('NN_model.h5')
    NN_probs = NN_model.predict(input_data_normalized)

    # Loading the saved XGBoost model from the file
    with open('XGB_model.pkl', 'rb') as file:
        XGB_model = pickle.load(file)
    XGB_probs = XGB_model.predict_proba(input_data_normalized)

    # Loading the saved RF model from the file
    with open('RF_model.pkl', 'rb') as file:
        RF_model = pickle.load(file)
    RF_probs = RF_model.predict_proba(input_data_normalized)

    # Collecting model predictions for validation data
    model_predictions = [gmm_probs, NN_probs, XGB_probs, RF_probs]
    
    # Assigning the best weights obtained from random search
    gmm_weight = best_weights[0]
    NN_weight = best_weights[1]
    XGB_weight = best_weights[2]
    RF_weight = best_weights[3]
    
    # Calculating final probabilities using the best weights for each model
    final_probs = (gmm_weight * gmm_probs + NN_weight * NN_probs +
                   XGB_weight * XGB_probs + RF_weight * RF_probs) / 4

    df = probabilities_to_dataframe(final_probs)
    return df











