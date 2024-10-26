import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def preprocess_text(text):
    text = re.sub(r'<[^>]*>', '', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    text = text.lower()  
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]
    return ' '.join(tokens)

emails = ["Free money!!!", "Hi, how are you?", "Win a free iPhone now!!!", "Let's catch up for lunch."]
labels = [1, 0, 1, 0]

preprocessed_emails = [preprocess_text(email) for email in emails]

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(preprocessed_emails)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
