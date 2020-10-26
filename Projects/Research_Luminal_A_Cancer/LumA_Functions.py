import time
import multiprocessing
from sklearn.linear_model import LogisticRegression
from functools import partial

	
def find_best_feature(selected_features, X_train, X_val, C_param, y_train, feature):
    # decide on which features we are using (selected_features + feature)
    features_in_use = selected_features + [feature]
    # print(features_in_use, int(feature))
    X_train_filt = X_train[:, features_in_use]
    X_val_filt = X_val[:, features_in_use]

    # Fit a Logistic regression model
    lr = LogisticRegression(C=C_param, random_state=0).fit(X_train_filt, y_train)
    # Get the accuracy rate using the validation set
    accu_train = lr.score(X_train_filt, y_train)
    # Tuple format = (feature number, training accuracy found)
    return (feature, accu_train)


def forward_selection(data, X_train, y_train, X_val, y_val, max_features, C_param=1):
    # Getting all the features and remembering to remove the isLumA column
    features = [i for i in range(data.shape[1] - 1)]
    selected_features = []
    selected_features_by_name = []
    highest_accu = 0
    best_feature = None
    selected_features_dict = []
    # Iterations for amount of features we want to select
    for i in range(max_features):
        # Keeping track of time, to monitor how long it takes for each iteration
        start = time.time()
        best_feature = None
        highest_accu = 0
        results = []
        # Create a threadpool of 5 process
        pool = multiprocessing.Pool(4)
        # Creating a dummy function in order to send multiple inputs to the function
        func = partial(find_best_feature, selected_features, X_train, X_val, C_param, y_train)
        # Gather the results
        result = pool.map(func, features)
        pool.close()
        pool.join()
        # Iterate through the results, find the best_feature
        for res in result:
            if res[1] > highest_accu:
                highest_accu = res[1]
                best_feature = res[0]

        # Add the best feature to the list
        selected_features.append(best_feature)
        selected_features_by_name.append(data.columns[best_feature])
        # Train the model again with these features
        X_train_filt = X_train[:, selected_features]
        X_val_filt = X_val[:, selected_features]
        lr = LogisticRegression(C=C_param, random_state=0).fit(X_train_filt, y_train)
        # Measure Validation accuracy with the features
        accu_val = lr.score(X_val_filt, y_val)
        # Remove the feature found from the list of features
        features.remove(best_feature)

        # Measure time
        times = time.time() - start
        # Populate dictionary of info and add it to our list and continue
        selected_features_dict.append({"Feature": selected_features_by_name.copy(), "Iteration": i + 1,
                                       "Training accuracy": highest_accu, "Validation Accuracy": accu_val,
                                       "Time": times})

    # return a dictionary
    return (selected_features_dict, highest_accu);
	
	


