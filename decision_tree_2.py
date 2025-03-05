#-------------------------------------------------------------------------
# AUTHOR: Johnny Nguyen
# FILENAME: decision_tree_2.py
# SPECIFICATION: This program creates, utilizes, and trains decision trees and evaluates their performance.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 6-7 hours approximately
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

# Converting data types from data to numerical beforehand
age_map = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
prescription_map = {'Myope': 1, 'Hypermetrope': 2}
astigmatism_map = {'Yes': 1, 'No': 2}
tear_map = {'Normal': 1, 'Reduced': 2}
class_map = {'Yes': 1, 'No': 2}

for ds in dataSets:
    dbTraining = []
    X = []
    Y = []

    # Read training data
    try:
        with open(ds, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0:  # Skip the header
                    dbTraining.append(row)
    except FileNotFoundError:
        print(f"Error: File {ds} not found.")
        continue

    # Check if training data is empty
    if not dbTraining:
        print(f"Error: No training data found in {ds}.")
        continue

    # Transform the original categorical training features to numbers and add to the 4D array X.
    # For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    # --> add your Python code here
    for row in dbTraining:
        try:
            age = age_map[row[0]]
            spectacle = prescription_map[row[1]]
            astigmatism = astigmatism_map[row[2]]
            tear = tear_map[row[3]]
            X.append([age, spectacle, astigmatism, tear])
    # Transform the original categorical training classes to numbers and add to the vector Y.
    # For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    # --> add your Python code here
            Y.append(class_map[row[4]])
        except KeyError as e:
            print(f"KeyError: Check the mappings for row {row} in {ds}. Missing key: {e}")
            exit()
        except IndexError:
            print(f"IndexError: Invalid row format in {ds}: {row}")
            exit()

    total_accuracy = 0.0

    #Loop your training and test tasks 10 times here
    for _ in range(10):
        #Fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf.fit(X, Y)  # Ensure X and Y are not empty

        # Read the test data and add this data to dbTest
        # --> add your Python code here
        dbTest = []
        try:
            with open('contact_lens_test.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                for i, row in enumerate(reader):
                    if i > 0:  # Skip the header
                        dbTest.append(row)
        except FileNotFoundError:
            print("Error: Test file 'contact_lens_test.csv' not found.")
            exit()

        # Check if test data is empty
        if not dbTest:
            print("Error: No test data found.")
            exit()

        correct_predictions = 0
        total_instances = len(dbTest)

        for data in dbTest:
            # Transform the features of the test instances to numbers following the same strategy done during training,
            # and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            # where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            # --> add your Python code here
            try:
                # Convert test features to numerical values
                age = age_map[data[0]]
                spectacle = prescription_map[data[1]]
                astigmatism = astigmatism_map[data[2]]
                tear = tear_map[data[3]]
                features = [age, spectacle, astigmatism, tear]

                # Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
                # --> add your Python code here
                predicted_class = clf.predict([features])[0]
                true_class = class_map[data[4]]
                if predicted_class == true_class:
                    correct_predictions += 1
            except KeyError as e:
                print(f"KeyError: Check test data mappings for {data}. Missing key: {e}")
                exit()
            except IndexError:
                print(f"IndexError: Invalid test data format: {data}")
                exit()

        # Find the average of this model during the 10 runs (training and test set)
        # --> add your Python code here
        accuracy = correct_predictions / total_instances if total_instances > 0 else 0
        total_accuracy += accuracy

    # Print the average accuracy of this model during the 10 runs (training and test set).
    # Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    # --> add your Python code here
    average_accuracy = total_accuracy / 10
    print(f"Final accuracy when training on {ds}: {average_accuracy:.1f}")
