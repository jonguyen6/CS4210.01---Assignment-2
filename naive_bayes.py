#-------------------------------------------------------------------------
# AUTHOR: Johnny Nguyen
# FILENAME: naive_bayes.py
# SPECIFICATION: This program predicts the most probable outcome based on naive bayes classification.
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

# Mapping the data to numerical values beforehand
outlook_mapping = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
temperature_mapping = {'Hot': 1, 'Mild': 2, 'Cool': 3}
humidity_mapping = {'High': 1, 'Normal': 2}
wind_mapping = {'Strong': 1, 'Weak': 2}
play_tennis_mapping = {'Yes': 1, 'No': 2}

#Reading the training data in a csv file
#--> add your Python code here
X = []
Y = []
with open('weather_training.csv', 'r') as train_file:
    reader = csv.reader(train_file)
    next(reader)  # Skip header
    for row in reader:
        outlook = outlook_mapping[row[1]]
        temperature = temperature_mapping[row[2]]
        humidity = humidity_mapping[row[3]]
        wind = wind_mapping[row[4]]
        play_tennis = play_tennis_mapping[row[5]]
        X.append([outlook, temperature, humidity, wind])
        Y.append(play_tennis)

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> I decided to just transform the original training features beforehand for simplicity.

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> Similarly, I decided to just transform the original training features beforehand for simplicity.

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
test_entries = []
with open('weather_test.csv', 'r') as test_file:
    reader = csv.reader(test_file)
    next(reader)  # Skip header
    for row in reader:
        day = row[0]
        outlook = row[1]
        temperature = row[2]
        humidity = row[3]
        wind = row[4]
        test_entries.append({
            'day': day,
            'outlook': outlook,
            'temperature': temperature,
            'humidity': humidity,
            'wind': wind,
            'features': [
                outlook_mapping[outlook],
                temperature_mapping[temperature],
                humidity_mapping[humidity],
                wind_mapping[wind]
            ]
        })

# Prepare to reverse the class mapping for output
reverse_play_tennis = {1: 'Yes', 2: 'No'}

#Printing the header os the solution
#--> add your Python code here
print("Day         Outlook     Temperature     Humidity     Wind     PlayTennis      Confidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for entry in test_entries:
    features = entry['features']
    probs = clf.predict_proba([features])[0]
    max_prob = max(probs)
    if max_prob >= 0.75:
        predicted_class_idx = probs.argmax()
        predicted_class = clf.classes_[predicted_class_idx]
        predicted_label = reverse_play_tennis[predicted_class]
        # Format the output to align columns
        print(f"{entry['day']: <12}{entry['outlook']: <12}{entry['temperature']: <16}{entry['humidity']: <12}{entry['wind']: <10}{predicted_label: <12}{max_prob:.2f}")