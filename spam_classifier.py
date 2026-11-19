# Step 1: Import required libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Step 2: Create training messages
messages = [
    # Original Spam messages
    "Congratulations you won a free prize",
    "Claim your free gift now",
    "You have won a lottery",
    "Click this link to get a free reward",
    "Win cash now congratulations",
    "Urgent offer claim your prize",
    "Free coupons available click now",
    "You are selected for a free gift",

    # New Spam messages
    "Earn money from home with no investment",
    "You have been selected for a cash reward",
    "Your bank account needs verification immediately",
    "Get an exclusive discount on your purchase",
    "You won an iPhone in our lucky draw",
    "Earn unlimited income online",
    "Claim your special bonus today",
    "Congratulations you received a cashback offer",

    # Original Not Spam messages
    "Meeting is scheduled for tomorrow",
    "Please submit the assignment today",
    "Can we meet after class",
    "Your lab record is ready",
    "Please call me when you are free",
    "Class starts at ten tomorrow",
    "The project meeting is at four",
    "Bring your notebook to class",

    # New Not Spam messages
    "I will reach the office by 10 AM",
    "Your interview is scheduled for Monday",
    "Please send me the project document",
    "Let's have lunch together this afternoon",
    "The team meeting starts at 11 AM",
    "I completed my homework yesterday",
    "Please share the class notes",
    "I will call you after reaching home"
]

# Step 3: Create labels for each message
labels = [
    # Original Spam labels
    "Spam", "Spam", "Spam", "Spam",
    "Spam", "Spam", "Spam", "Spam",

    # New Spam labels
    "Spam", "Spam", "Spam", "Spam",
    "Spam", "Spam", "Spam", "Spam",

    # Original Not Spam labels
    "Not Spam", "Not Spam", "Not Spam", "Not Spam",
    "Not Spam", "Not Spam", "Not Spam", "Not Spam",

    # New Not Spam labels
    "Not Spam", "Not Spam", "Not Spam", "Not Spam",
    "Not Spam", "Not Spam", "Not Spam", "Not Spam"
]
# Step 4: Split dataset into training and testing data
X_train_text, X_test_text, y_train, y_test = train_test_split(
    messages,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

# Step 5: Convert text into numerical data
vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# Step 6: Create and train the model
model = MultinomialNB()
model.fit(X_train, y_train)

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "spam_vectorizer.pkl")
print("Model and vectorizer saved successfully")
# Step 7: Accept messages from the user

print("\n================================")
print("     SPAM MESSAGE CLASSIFIER")
print("================================")
print("Type 'exit' to stop the program.")

while True:

    msg = input("\nEnter your message: ")

    # Check whether the user wants to exit
    if msg.lower() == "exit":
        print("Thank you!")
        break

    # Check for empty messages
    if not msg.strip():
        print("Please enter a message.")
        continue

    # Convert the entered message into numerical data
    sample = vectorizer.transform([msg])

    # Predict the category
    prediction = model.predict(sample)[0]

    # Calculate confidence
    confidence = max(model.predict_proba(sample)[0]) * 100

    # Display the result
    print("Result     :", prediction)
    print("Confidence : {:.2f}%".format(confidence))