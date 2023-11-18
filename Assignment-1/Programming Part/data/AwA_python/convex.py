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

    # Step-1: Calculating the mean of each class seen.
    mean_encountered = np.array([np.mean(class_seen , axis = 0) for class_seen in X_seen])

    # Step-2: Calculating the similarity of each unseen class (10 in number) with each seen class (40 in number)
    similarity_product = np.empty((10, 40))
    # similarity_product[i][j]: Similarity between the (i + 1)th unseen class and the (j + 1)th seen class.
    """
    Storing in a matrix form is more handy, as in the end, we can just multiply the \mu vector with this similarity matrix
    and we shall calculate all the unseen means in a single step.
    """
    for class_unseen in range(10):
        for class_seen in range(40):
            similarity_product[class_unseen][class_seen] = np.dot(class_attributes_unseen[class_unseen], class_attributes_seen[class_seen])
        # Step-3: Normalising all the individual vectors to 1. 
        overall_sum = np.sum(similarity_product[class_unseen])
        for class_seen in range(40):
            similarity_product[class_unseen][class_seen] /= overall_sum

    # Step-4: Computing the mean of each unseen class using a convex combination of the means of seen classes.
    """
    This step now becomes easy, as we can directly multiply the similarity matric with the mean vector, and get all the 
    unencountered mean vectors, in a single go!
    """
    mean_unseen = np.matmul(similarity_product, mean_encountered)

    return mean_unseen, Xtest, Ytest

# Step-5: Applying the model to predict labels on unseen class' test input.
def predict_label(mean_unseen, test_input):
    min_distance = sys.maxsize
    position = -1
    index = 0
    for mean_vector in mean_unseen:
        curr_distance = np.linalg.norm(test_input - mean_vector)
        if curr_distance <= min_distance:
            min_distance = curr_distance
            position = index
        index += 1
    return position + 1

# Driver Function.
def main():
    # Training the model.
    mean_unseen, Xtest, Ytest = training_the_model()
    # Step-6: Computing the classification accuracy as the (no. of correctly classified points) / (total number of testing points)
    n = Ytest.shape[0]
    count = 0
    for index in range(n):
        if predict_label(mean_unseen, Xtest[index]) == Ytest[index]:
            count += 1
    print("A total of", n, "testing points were considered.")
    print("Accuracy achieved on those", n, "testing points:", (count * 100) / n, "%.")

if __name__ == "__main__":
    main()