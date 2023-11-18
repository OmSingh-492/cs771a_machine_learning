# Importing the required modules.
import sys
import numpy as np

def training_the_model():
    # Loading the dataset.
    X_seen = np.load('X_seen.npy', encoding = 'bytes', allow_pickle = True)
    Xtest = np.load('Xtest.npy', encoding = 'bytes', allow_pickle = True)
    Ytest = np.load('Ytest.npy', encoding = 'bytes', allow_pickle = True)
    class_attributes_seen = np.load('class_attributes_seen.npy', encoding = 'bytes', allow_pickle = True)
    class_attributes_unseen = np.load('class_attributes_unseen.npy', encoding = 'bytes', allow_pickle = True)

    # Step-1: Calculating the mean of all the seen classes.
    mean_encountered = np.array([np.mean(class_seen, axis = 0) for class_seen in X_seen])
    """We store all the lambda values which have to be checked for, 
    so that we can just collect them from a loop, and run our regression model on that."""
    hyperparam = np.array([0.01, 0.1, 1, 10, 20, 50, 100])

    # Step-2: Learning the Weight matrix, and hence all the weight vectors at once.
    """
    For this part, we shall first evaluate (A_s^T A_s + \lambda I) in one go for all the cases, 
    and then calculate its inverse. We store them in a variable term, and call the final weight matrix W.
    curr_val: Current value of the hyperparameter lambda.
    """
    term = np.array([(np.matmul(class_attributes_seen.transpose(), class_attributes_seen) + curr_val * np.eye(85)) for curr_val in hyperparam])
    W = np.array([np.matmul(np.linalg.inv(matrix), np.matmul(class_attributes_seen.transpose(), mean_encountered)) for matrix in term])
    
    # Step-3: Calculate the means of all the unseen classes.
    mean_unseen = np.array([np.matmul(class_attributes_unseen, weight) for weight in W])
    return mean_unseen, Xtest, Ytest, hyperparam

# Step-5: Applying the model to predict labels on unseen class' test input.
def predict_label(mean_unseen, test_input, curr_index):
    min_distance = sys.maxsize
    position = -1
    index = 0
    for mean_vector in mean_unseen[curr_index]:
        curr_distance = np.linalg.norm(test_input - mean_vector)
        if curr_distance <= min_distance:
            min_distance = curr_distance
            position = index
        index += 1
    return position + 1

# Driver Function.
def main():
    # Training the model.
    mean_unseen, Xtest, Ytest, hyperparam = training_the_model()
    size = hyperparam.shape[0]
    n = Ytest.shape[0]
    """
    Step-6: Computing the classification accuracy as the 
    (no. of correctly classified points) / (total number of testing points) for each lambda. """
    print("A total of", n, "testing points were considered.")
    for curr_index in range(size):
        count = 0
        for index in range(n):
            if predict_label(mean_unseen, Xtest[index], curr_index) == Ytest[index]:
                count += 1
        print("For lambda =", hyperparam[curr_index], "accuracy achieved on the testing points =", (count * 100) / n, "%.")

if __name__ == "__main__":
    main()