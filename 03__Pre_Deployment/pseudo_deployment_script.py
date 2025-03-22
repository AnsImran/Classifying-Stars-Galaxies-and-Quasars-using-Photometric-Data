# https://drive.google.com/drive/folders/1dL39eCvXAzXnqAcw2n_wBz1ou3xDFRdO
from fastapi import FastAPI
from pydantic import BaseModel
import pickle


import pandas as pd
import numpy as np


from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

# for aws lambda
from mangum import Mangum

from fastapi.responses import JSONResponse
import uvicorn







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

    # # Loading the saved GMM model from the file
    # with open('gmm_model.pkl', 'rb') as file:
    #     gmm_model = pickle.load(file)
    gmm_probs = gmm_model.predict_proba(input_data_normalized)

    # # Loading the saved NN model
    # NN_model = load_model('NN_model.h5')
    NN_probs = NN_model.predict(input_data_normalized)

    # # Loading the saved XGBoost model from the file
    # with open('XGB_model.pkl', 'rb') as file:
    #     XGB_model = pickle.load(file)
    XGB_probs = XGB_model.predict_proba(input_data_normalized)

    # # Loading the saved RF model from the file
    # with open('RF_model.pkl', 'rb') as file:
    #     RF_model = pickle.load(file)
    RF_probs = RF_model.predict_proba(input_data_normalized)

    # # Collecting model predictions for validation data
    model_predictions = [gmm_probs, NN_probs, XGB_probs, RF_probs]
    # model_predictions = [NN_probs]
    
    # Assigning the best weights obtained from random search
    gmm_weight = best_weights[0]
    NN_weight = best_weights[1]
    XGB_weight = best_weights[2]
    RF_weight = best_weights[3]
    # NN_weight = 1

    # # Calculating final probabilities using the best weights for each model
    final_probs = (gmm_weight * gmm_probs + NN_weight * NN_probs +
                   XGB_weight * XGB_probs + RF_weight * RF_probs) / 4
    # final_probs = (NN_weight * NN_probs) / 1

    df = probabilities_to_dataframe(final_probs)
    return df




# u, g, r, i, z, rs - the order of features


# # some Quasars Examples - 0
# {
#     "ultraviolet_filter" : 23.48827,
#     "green_filter"       : 23.33776,
#     "red_filter"         : 21.32195,
#     "near_infrared"      : 20.25615,
#     "infrared_filter"    : 19.54544,
#     "red_shift"          : 1.424659
# }
# 23.48827 	23.33776 	21.32195 	20.25615 	19.54544 	1.424659
# 21.46973 	21.17624 	20.92829 	20.60826 	20.42573 	0.586455
# 20.38562 	20.40514 	20.29996 	20.05918 	19.89044 	2.031528


# # some galaxies examples - 1
# {
#     "ultraviolet_filter" : 25.44159,
#     "green_filter"       : 23.04573,
#     "red_filter"         : 21.73499,
#     "near_infrared"      : 20.65421,
#     "infrared_filter"    : 19.81532,
#     "red_shift"          : 0.634074
# }
# 25.44159 	23.04573 	21.73499 	20.65421 	19.81532 	0.634074
# 22.74218 	21.81012 	20.00882 	19.08053 	18.65647 	0.570393
# 19.60586 	18.17074 	17.27391 	16.83142 	16.51841 	0.135258


# # some stars examples - 2
# {
#     "ultraviolet_filter" : 18.97704,
#     "green_filter"       : 18.10361,
#     "red_filter"         : 17.84162,
#     "near_infrared"      : 17.75269,
#     "infrared_filter"    : 17.74013,
#     "red_shift"          : 0.000067
# }
# 18.97704 	18.10361 	17.84162 	17.75269 	17.74013 	0.000067
# 17.76627 	16.77135 	16.46240 	16.35024 	16.33692 	-0.000143
# 17.70832 	16.60293 	16.21125 	16.04970 	15.98337 	-0.000172


# loading means for each feature
file_path = 'mean.csv'
mean      = np.loadtxt(file_path, delimiter=',')
print(mean)

# Loading standard deviations for each feature
file_path = 'standard_deviation.csv'
std       = np.loadtxt(file_path, delimiter=',')
print(std)

# Loading best weights from all models
file_path = 'best_weights.csv'
weights   = np.loadtxt(file_path, delimiter=',')
weights





# # # loading the saved models
# Loading the saved gaussian mixture model
with open('gmm_model.pkl', 'rb') as file:
    gmm_model = pickle.load(file)

# Loading the saved NN model
NN_model = load_model('NN_model.keras')

# Loading the saved XGBoost model
with open('XGB_model.pkl', 'rb') as file:
    XGB_model = pickle.load(file)

# Loading the saved RF model from the file
with open('RF_model.pkl', 'rb') as file:
    RF_model = pickle.load(file)





app     = FastAPI()
handler = Mangum(app)

# # Features & what do they mean?
# u 	Ultraviolet filter in the photometric system
# g 	Green filter in the photometric system
# r 	Red filter in the photometric system
# i 	Near Infrared filter in the photometric system
# z     Infrared filter in the photometric system
# rs    Red shift

class Model_input(BaseModel):    
    ultraviolet_filter : float
    green_filter       : float
    red_filter         : float
    near_infrared      : float
    infrared_filter    : float
    red_shift          : float




@app.post('/predict')
def stellar_pred(input_parameters : Model_input):

    u  = input_parameters.ultraviolet_filter
    g  = input_parameters.green_filter
    r  = input_parameters.red_filter
    i  = input_parameters.near_infrared
    z  = input_parameters.infrared_filter
    rs = input_parameters.red_shift
    
    input_array = np.array([u, g, r, i, z, rs]).reshape(1,6)
    input_array = input_array - mean
    input_array = input_array/std
    
    df = ensemble_of_models_without_true_labels(input_array, weights)
#    print(df)
    return JSONResponse({"prediction": df.iloc[0].to_dict()})


if __name__=="__main__":
  uvicorn.run(app,host="0.0.0.0",port=9000)







