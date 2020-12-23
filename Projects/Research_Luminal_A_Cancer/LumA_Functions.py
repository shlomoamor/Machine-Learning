import time
import multiprocessing
from sklearn.linear_model import LogisticRegression
from functools import partial
import pandas as pd
import numpy as np

"""
Inputs: 
- data: in numpy form
- bins: used for binning, default setting is 5 bins
Outputs:
- data input as ranked and binned.
Description:
This function returns the data input in ranked and binned
"""
def bin_rank(data, bins=5):
    
    data = pd.DataFrame(data)
    rank_data = data.rank(axis=1)
    
#     for i, row in rank_data.iterrows():
#         rank_data.loc[i, :] = pd.cut(row, bins=bins, labels=False)
    
    return rank_data.values

"""
Inputs: 
- X_train, y_train:  for training the model
- X_val, y_val: for validating selected feature
- feature:  feature to be added to the selected set and check accuracy
- selected_features(_by_name): the start to our feature selection list.
Outputs:
- (feature, accu_train): returns the training accuracy with the feature added to the selected set
Description:
This function receives a feature and a selected set and trains the model according to that set and returns the training accuracy. This function is used as part of deciding the best feature for the current iteration.
"""
def find_best_feature(selected_features, X_train, C_param, y_train, feature):
    # decide on which features we are using (selected_features + feature)
    features_in_use = np.append(selected_features,feature)
    X_train_filt = X_train[:, features_in_use]
    X_train_filt_ranked = bin_rank(X_train_filt)
    
    # Fit a Logistic regression model
    lr = LogisticRegression(C=C_param, random_state=0).fit(X_train_filt_ranked, y_train)
    # Get the accuracy rate using the validation set
    accu_train = lr.score(X_train_filt_ranked, y_train)
    # Tuple format = (feature number, training accuracy found)
    return (feature, accu_train)

"""
Inputs: 
- data: for getting feature names by index
- X_train, y_train:  for training the model
- X_val, y_val: for validating selected feature
- max_rounds:  to cap the number of rounds we forward select
- best_feature: already found best feature to add into our set
- selected_features(_by_name): the start to our feature selection list.
Outputs:
- best feature, selected sets and boolean flag whether set has converged
Description:
This function adds the best feature found in the iteration of forward selection and then iterativly checks which single feature removal will lead to a max accuracy. If the sets remain unchanged then we have converged and therefore from now on this function is redundant and we will simply add best feature found to our selected set.
"""
def selected_feature_check(data, X_train, y_train, selected_features, selected_features_by_name, C_param, best_feature):
    # Save the previous selected features to check later if we have made a change
    prev_selected_features = selected_features.copy()
    prev_selected_features_by_name = selected_features_by_name.copy()
    # Add the best feature to the list
    selected_features = np.append(selected_features,best_feature)
    selected_features_by_name.append(data.columns[best_feature])
    feature_removal_score = {}
    
    # Iterate through each feature in the list and remove it
    for feature in selected_features:
        temp_features =  np.setdiff1d(selected_features,np.array([feature]))
        X_train_filt = X_train[:, temp_features]
        X_train_filt_ranked = bin_rank(X_train_filt)
        # Fit a Logistic regression model
        lr = LogisticRegression(C=C_param, random_state=0).fit(X_train_filt_ranked, y_train)
        # Get the accuracy rate using the validation set
        accu_train = lr.score(X_train_filt_ranked, y_train)
        feature_removal_score[feature] = accu_train
    
    # Get the feature which causes the highest accuracy without it
    max_key = max(feature_removal_score, key=lambda k: feature_removal_score[k])
    selected_features = np.setdiff1d(selected_features, np.array([max_key]))
    max_key_name = data.columns[max_key]
    selected_features_by_name = list(set(selected_features_by_name) - set([max_key_name]))
    
    # Check if we have made any changes, if not let the caller know that this function is no longer needed
    if np.array_equal(selected_features,prev_selected_features):
        return max_key,selected_features, selected_features_by_name, True
    else:
        return max_key,selected_features, selected_features_by_name, False
   
    
"""
Inputs: 
- data: for getting feature names by index
- X_train, y_train:  for training the model
- X_val, y_val: for validating selected feature
- max_rounds:  to cap the number of rounds we forward select
- thread_pool: how many threads we should use in our pool
- selected_features(_by_name): the start to our feature selection list.
Outputs:
-selected_features_dict: contains all the information for every round of each subset of features at every adding interval and time it took to run
Description:
This function preforms forward feature selection, starting with an input of 20 features, we replace the 20 features and continue to select the best predictive features up until we reach max_rounds
"""
def forward_selection(data, X_train, y_train, X_val, y_val, max_rounds, thread_Pool,selected_features =[],
                      selected_features_by_name = [], C_param=1):
    # Getting all the features and remembering to remove the isLumA column
    features = [i for i in range(data.shape[1] - 1)]
    # Removing from the available features what we already have in the selected features
    features = np.setdiff1d(features, selected_features)
    # Maaking sure that are selected features contain what we already have
    selected_features = selected_features
    selected_features_by_name = selected_features_by_name
    highest_accu = 0
    best_feature = None
    selected_features_dict = []
    # Boolean flag indicating whether we are to continue trying to swap out the 20 original features
    isFinishedSwapping = False
    
    # IIterations for amount of rounds we want to prefrorm (Note: max_rounds != number of features we will have at the end)
    for i in range(max_rounds):
        # Keeping track of time, to monitor how long it takes for each iteration
        start = time.time()
        best_feature = None
        highest_accu = 0
        result = []
        # Create a thread pool of thread_Pool amount of process
        pool = multiprocessing.Pool(thread_Pool)
        # Creating a dummy function in order to send multiple inputs to the function
        func = partial(find_best_feature, selected_features, X_train, C_param, y_train)
        # Gather the results
        result = pool.map(func, features)
        pool.close()
        pool.join()
        #Iterate through the results, find the best_feature
        for res in result:
            if res[1] > highest_accu:
                highest_accu = res[1]
                best_feature = res[0]
        
        # Add the best feature to the list
        if not isFinishedSwapping:
            # We still havent converged in our starting selected features, so we run the function to updated the selected features
            best_feature,selected_features, selected_features_by_name, isFinishedSwapping = selected_feature_check(
                data,X_train, y_train,selected_features, selected_features_by_name, C_param,best_feature)
        else:
            # We have converged and now just the best feature
            selected_features = np.append(selected_features,best_feature)
            selected_features_by_name.append(data.columns[best_feature])
        
        # Train the model again with these selected features
        X_train_filt = X_train[:, selected_features]
        X_val_filt = X_val[:, selected_features]
        # Rank the data
        X_train_filt_ranked = bin_rank(X_train_filt)
        X_val_filt_ranked = bin_rank(X_val_filt)
        # Train the model
        lr = LogisticRegression(C=C_param, random_state=0).fit(X_train_filt_ranked, y_train)
        # Measure Validation accuracy with the features
        accu_val = lr.score(X_val_filt_ranked, y_val)
        # Remove the feature found from the list of features
        features = np.setdiff1d(features, np.array([best_feature]))
        # Measure time
        times = time.time() - start
        
        # Populate dictionary of info and add it to our list and continue
        selected_features_dict.append({"Feature": selected_features_by_name.copy(), "Iteration": i + 1,
                                       "Training accuracy": highest_accu, "Validation Accuracy": accu_val,"Time": times})
        print("Round:",i,"Features found so far are: ", selected_features_by_name, "\n")
    # return a dictionary
    return (selected_features_dict, highest_accu);