# STEP 1: Load Dataset
import pandas as pd

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

print("Dataset loaded")
print(df.head())
print(df['label'].value_counts())

# STEP 2: Preprocessing — Just lowercase and basic cleanup (no nltk)
df['cleaned_message'] = df['message'].str.lower().str.replace('[^a-zA-Z]', ' ', regex=True)

# View cleaned output
print("\nSample cleaned text:")
print(df[['message', 'cleaned_message']].head())

#step 3
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')  # built-in stopwords

# Fit and transform the cleaned messages
X = vectorizer.fit_transform(df['cleaned_message'])

# Show feature shape
print("TF-IDF feature matrix shape:", X.shape)


from sklearn.preprocessing import LabelEncoder

# Convert labels to 0 and 1
le = LabelEncoder()
y = le.fit_transform(df['label'])  # ham=0, spam=1

print("Labels encoded:", list(le.classes_))  # ['ham', 'spam']

#step 4
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 3: Make predictions
y_pred = model.predict(X_test)

# Step 4: Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
