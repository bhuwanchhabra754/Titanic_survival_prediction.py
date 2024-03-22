from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import time

# Load Titanic dataset (replace my dataset path with your dataset path)
data = pd.read_csv(r"/content/train.csv")

# Features (X) and target variable (y)
X = data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = data['Survived']

# Convert categorical variables to numerical using .loc
X.loc[:, 'Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X.loc[:, 'Embarked'] = X['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Handle missing values
X = X.dropna()
y = y.loc[X.index]

# Ensure X and y have the same number of samples

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # we split data into 80,20 percent 

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()

# Train the model
dt_classifier.fit(X_train, y_train)

# Evaluate the model on the training set
train_predictions = dt_classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)

# Evaluate the model on the testing set
test_predictions = dt_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

# Print model evaluation metrics
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("\nClassification Report for Testing Set:\n", classification_report(y_test, test_predictions))

# Function to predict survival and display result using pyautogui
def predict_survival():
    try:
        pclass = int(input("Enter Pclass: "))
        sex = input("Enter Sex (male/female): ")
        age = float(input("Enter Age: "))
        fare = float(input("Enter Fare: "))
        embarked = input("Enter Embarked (S=0/C=1/Q=2): ")

        # Make prediction
        prediction = dt_classifier.predict([[pclass, sex, age, fare, embarked]])

        # Display result
        if prediction[0] == 1:
            print("Survived")
        else:
            print("Not Survived")

    except ValueError:
        print("Invalid input. Please enter valid values.")

# Call the predict_survival function
predict_survival()
