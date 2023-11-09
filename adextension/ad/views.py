from django.shortcuts import render
from django.http import JsonResponse  # Import JsonResponse to send JSON responses
from django.views.decorators.csrf import csrf_exempt
from django.utils.datastructures import MultiValueDictKeyError


import re
import requests
from bs4 import BeautifulSoup
import json
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import FreqDist, ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
def main(url):
  from nltk.corpus import stopwords
  import numpy as np
  def extract_information(url):
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
    else:
        print("Failed to retrieve the webpage:", response.status_code)
        return None

    soup = BeautifulSoup(html_content, "html.parser")

    # Extract Title
    title = soup.title.string if soup.title else ""

    # Extract Meta Description
    meta_description = soup.find("meta", attrs={"name": "description"})
    meta_description = meta_description["content"] if meta_description else ""

    # Extract Header Tags
    header_tags = [header.text.strip() for header in soup.find_all(["h1", "h2", "h3"])]

    # Extract Text Content
    text_content = soup.get_text(strip=True)

    # Extract Images
    images = [image['src'] for image in soup.find_all('img')]

    # Extract Links
    links = [link['href'] for link in soup.find_all('a', href=True)]

    # Extract Contact Information
    contact_info = re.findall(r'(\+\d{1,3}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', html_content)
    contact_info = [re.sub(r'[^\d-]', '', info) for info in contact_info]

    # Create a dictionary to store the extracted information
    data = {
        "url": url,
        "title": title,
        "meta_description": meta_description,
        "header_tags": header_tags,
        "text_content": text_content,
        "images": images,
        "links": links,
        "contact_info": contact_info
    }

    return data


  def save_to_json(data, output_file):
      with open(output_file, "w") as json_file:
          json.dump(data, json_file, indent=4)
      #print("Data saved to", output_file)


  # Example usage
  #url = input("Enter the Url").strip()
  output_file = "website_data.json"

  extracted_data = extract_information(url)
  if extracted_data:
      save_to_json(extracted_data, output_file)

    # Define the path to the JSON file
  json_file = "website_data.json"

  # Load the JSON data
  with open(json_file) as file:
      data = json.load(file)



    # Load the JSON data
  json_file = "website_data.json"
  with open(json_file) as file:
      data = json.load(file)

  # Preprocessing
  def clean_text(text):
      # Remove HTML tags
      cleaned_text = re.sub('<[^<]+?>', '', text)
      # Remove special characters and numbers
      cleaned_text = re.sub('[^a-zA-Z]', ' ', cleaned_text)
      # Convert to lowercase
      cleaned_text = cleaned_text.lower()
      # Tokenize the text
      tokens = word_tokenize(cleaned_text)
      # Remove stopwords
      from nltk.corpus import stopwords
      stop_words = set(stopwords.words('english'))
      tokens = [word for word in tokens if word not in stop_words]
      # Lemmatize the tokens
      lemmatizer = WordNetLemmatizer()
      tokens = [lemmatizer.lemmatize(token) for token in tokens]
      # Join the tokens back to a single string
      cleaned_text = ' '.join(tokens)
      return cleaned_text

  # Apply data preprocessing
  cleaned_content = clean_text(data['text_content'])
  cleaned_title = clean_text(data['title'])
  cleaned_meta_description = clean_text(data['meta_description'])
  # NLP Analysis


  # Tokenization
  tokens = word_tokenize(cleaned_content)
  #print("Tokens:", tokens)

  # Sentence Tokenization
  sentences = sent_tokenize(cleaned_content)
  #print("Sentences:", sentences)

  # Part-of-Speech (POS) Tagging
  pos_tags = nltk.pos_tag(tokens)
  #print("POS Tags:", pos_tags)

  # Named Entity Recognition (NER)
  nltk.download('maxent_ne_chunker')
  nltk.download('words')
  ner_tags = nltk.ne_chunk(pos_tags)
  #print("NER Tags:", ner_tags)

  # Sentiment Analysis (example using VaderSentiment)
  from nltk.sentiment import SentimentIntensityAnalyzer

  analyzer = SentimentIntensityAnalyzer()
  sentiment_scores = analyzer.polarity_scores(cleaned_content)
  #print("Sentiment Scores:", sentiment_scores)

    # Define the number of words in a phrase
  phrase_length = 2

  # Define stopwords to be ignored during phrase extraction
  stopwords = set(stopwords.words('english'))

  # Extract keywords
  keywords = word_tokenize(cleaned_content)
  keyword_freq = FreqDist(keywords)
  top_keywords = keyword_freq.most_common(10)  # Extract top 10 most frequent keywords

  # Extract important phrases
  phrase_freq = FreqDist()
  phrases = ngrams(tokens, phrase_length)
  for phrase in phrases:
      if all(word not in stopwords for word in phrase):
          phrase_freq[tuple(phrase)] += 1

  top_phrases = phrase_freq.most_common(10)  # Extract top 10 important phrases

  # Extract named entities (remaining code remains the same)
  entities = [entity for entity in ner_tags if hasattr(entity, 'label')]
  named_entities = [ne[0] for entity in entities for ne in entity.leaves()]

  # Create a dictionary to store the extracted features
  extracted_features = {
      "Top Keywords": top_keywords,
      "Top Phrases": top_phrases,
      "Named Entities": named_entities
  }

  # Specify the file path
  output_file_path = "output.json"

  # Save the dictionary as JSON
  with open(output_file_path, "w") as f:
      json.dump(extracted_features, f)

  #print("Output data saved to:", output_file_path)

    # Define the path to the JSON file
  json_file = "output.json"

  # Load the JSON data
  with open(json_file) as file:
      data = json.load(file)

  # Print the loaded data
  #print(data)

    # Read data from JSON file
  with open('output.json', 'r') as file:
      data = json.load(file)

  # Extract keywords and phrases from the data
  keywords = data['Top Keywords']
  phrases = data['Top Phrases']

  # Flatten the list of keywords and phrases
  keywords = [word for sublist in keywords for word in sublist]
  phrases = [phrase for sublist in phrases for phrase in sublist]

  # Convert the keywords and phrases to string type
  keywords = [str(word) for word in keywords]
  phrases = [str(phrase) for phrase in phrases]

  # Combine keywords and phrases
  combined_data = keywords + phrases

  # Create a TF-IDF vectorizer
  vectorizer = TfidfVectorizer()

  # Fit and transform the combined data to obtain the TF-IDF representation
  tfidf_matrix = vectorizer.fit_transform(combined_data)

  # Get the feature names (keywords) from the vectorizer
  feature_names = vectorizer.get_feature_names_out()

  # List to store documents with keywords
  documents_with_keywords = []

  # Iterate over the documents
  for i, doc in enumerate(combined_data):
      feature_index = tfidf_matrix[i, :].nonzero()[1]
      if len(feature_index) > 0:
          # Document contains keywords, add it to the list
          documents_with_keywords.append(doc)

  # Update combined_data with the filtered documents
  combined_data = documents_with_keywords

  # Update tfidf_matrix with the filtered documents
  tfidf_matrix = vectorizer.transform(combined_data)

  # Get the feature names (keywords) from the vectorizer
  feature_names = vectorizer.get_feature_names_out()

  # List to store the top keywords for each document
  top_keywords_per_document = []

  # Print the top keywords with highest TF-IDF scores
  num_keywords = 10  # Number of top keywords to extract
  for i, doc in enumerate(combined_data):
      feature_index = tfidf_matrix[i, :].nonzero()[1]
      tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
      sorted_tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
      top_keywords = [feature_names[i] for i, _ in sorted_tfidf_scores[:num_keywords]]
      top_keywords_per_document.append(top_keywords)

  # Save the output to a JSON file
  output = {'Top Keywords per Document': top_keywords_per_document}

  with open('keyword_extraction_output.json', 'w') as file:
      json.dump(output, file)


    # Specify the path to your JSON file
  json_file_path = 'keyword_extraction_output.json'  # Replace with the actual path to your JSON file

  # Read the JSON file
  with open(json_file_path, 'r') as file:
      json_data = json.load(file)

  json_data['ad_extensions'] = [
      {
        "extension_id": 1,
        "extension_text": "Discover the best deals at our website!",
        "category":"site extensions"
      },
      {
        "extension_id": 2,
        "extension_text": "Upgrade your experience at our place",
        "category":"location extension"
      },
      {
        "extension_id": 3,
        "extension_text": "Experience our latest collection. call now!",
        "category":"call extension"
      },
      {
        "extension_id": 4,
        "extension_text": "Upgrade your gaming experience with our latest collection. Buy now!",
        "category":"callout extension"
      },
      {
        "extension_id": 5,
        "extension_text": "Check out the prices of our products. Buy now!",
        "category":"Price extension"
      },
      {
        "extension_id": 6,
        "extension_text": "check out our app. download now!",
        "category":"App extension"
      },
      {
        "extension_id": 7,
        "extension_text": "Explore all our services",
        "category":"service extensions"
      }
    ]
  with open(json_file_path, 'w') as file:
      json.dump(json_data, file, indent=4)

        # Define the path to the JSON file
  json_file = "keyword_extraction_output.json"

  # Load the JSON data
  with open(json_file) as file:
      data = json.load(file)

  # Print the loaded data
  #print(data)

  # Read data from JSON file
  with open('keyword_extraction_output.json', 'r') as file:
      data = json.load(file)

  # Extract ad extensions and keywords from the data
  ad_extensions = data['ad_extensions']
  keywords = data['Top Keywords per Document']

  # Prepare the data in the desired format for training the model
  X = keywords  # Input features (keywords)
  y = ad_extensions  # Output labels (ad extensions)


  # Further processing or splitting into training/testing sets can be done as needed

    # Read data from JSON file
  with open('keyword_extraction_output.json', 'r') as file:
      data = json.load(file)

  # Extract keywords from the data
  keywords = data['Top Keywords per Document']

  # Convert the keywords to string format
  keywords = [' '.join(keyword) for keyword in keywords]

  # Create a CountVectorizer to convert keywords into a bag-of-words representation
  vectorizer = CountVectorizer()

  # Fit and transform the keywords to obtain the bag-of-words representation
  features = vectorizer.fit_transform(keywords)
  

  # Convert the bag-of-words representation to a numerical feature matrix
  feature_matrix = features.toarray()


    # Read data from JSON file
  with open('keyword_extraction_output.json', 'r') as file:
      data = json.load(file)

  # Add the feature matrix to the data
  data['Feature Matrix'] = feature_matrix.tolist()

  # Save the updated data to the JSON file
  with open('keyword_extraction_output.json', 'w') as file:
      json.dump(data, file, indent=4)

    # Define the path to the JSON file
  json_file = "keyword_extraction_output.json"

  # Load the JSON data
  with open(json_file) as file:
      data = json.load(file)



    # Read the ad_extensions_data.json file
  with open('keyword_extraction_output.json', 'r') as file:
      data = json.load(file)

  # Extract the ad_extensions from the data
  ad_extensions = data["ad_extensions"]

  # Extract the category labels from the ad_extensions
  categories = [extension["category"] for extension in ad_extensions]

  # Perform label encoding
  label_encoder = LabelEncoder()
  encoded_labels = label_encoder.fit_transform(categories)

  # Update the ad_extensions with the encoded labels
  for i, extension in enumerate(ad_extensions):
      extension["category_encoded"] = int(encoded_labels[i])

  # Save the updated ad_extensions_data.json file to a new file
  output_file = 'ad_extensions_data_encoded.json'
  with open(output_file, 'w') as file:
      json.dump(data, file, indent=4)

    # Define the path to the JSON file
  json_file = "ad_extensions_data_encoded.json"

  # Load the JSON data
  with open(json_file) as file:
      data = json.load(file)


    # Load the data
  with open('ad_extensions_data_encoded.json', 'r') as file:
      data = json.load(file)

  # Extract the features and labels
  keywords = data["Feature Matrix"]
  ad_extensions = data["ad_extensions"]

  # Check the number of samples in each data
  num_samples_keywords = len(keywords)
  num_samples_ad_extensions = len(ad_extensions)

  # Determine the minimum number of samples
  min_num_samples = min(num_samples_keywords, num_samples_ad_extensions)

  # Trim the data to have the same number of samples
  keywords = keywords[:min_num_samples]
  ad_extensions = ad_extensions[:min_num_samples]

  # Extract the labels
  labels = [extension["category_encoded"] for extension in ad_extensions]

  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(keywords, labels, test_size=0.2, random_state=42)

  # Convert the lists to numpy arrays
  X_train = np.array(X_train)
  X_test = np.array(X_test)
  y_train = np.array(y_train)
  y_test = np.array(y_test)

    # Load the data
  with open('ad_extensions_data_encoded.json', 'r') as file:
      data = json.load(file)

  # Extract the features and labels
  keywords = data["Feature Matrix"]
  ad_extensions = data["ad_extensions"]

  # Check the number of samples in each dataset
  num_samples_keywords = len(keywords)
  num_samples_ad_extensions = len(ad_extensions)

  # Determine the minimum number of samples
  min_num_samples = min(num_samples_keywords, num_samples_ad_extensions)

  # Trim the data to have the same number of samples
  keywords = keywords[:min_num_samples]
  ad_extensions = ad_extensions[:min_num_samples]

  # Extract the labels
  labels = [extension["category_encoded"] for extension in ad_extensions]

  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(keywords, labels, test_size=0.2, random_state=42)

  # Convert the lists to numpy arrays
  X_train = np.array(X_train)
  X_test = np.array(X_test)
  y_train = np.array(y_train)
  y_test = np.array(y_test)

  # Define the random forest classifier
  rf = RandomForestClassifier()

  # Define the parameter grid for hyperparameter tuning
  param_grid = {
      'n_estimators': [50, 100, 150],
      'max_depth': [None, 5, 10],
      'min_samples_split': [2, 5, 10]
  }

  # Create GridSearchCV object with LeaveOneOut
  grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=LeaveOneOut(), scoring='accuracy')

  # Perform grid search cross-validation
  grid_search.fit(X_train, y_train)

  # Get the best model from grid search
  best_model = grid_search.best_estimator_

  # Evaluate the best model on the test set
  test_accuracy = best_model.score(X_test, y_test)
  #print("Test Accuracy:", test_accuracy)

  from keras.models import Sequential
  from keras.layers import LSTM, Dense

  # Reshape the input arrays
  X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
  X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

  # Define the model architecture
  model = Sequential()
  model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
  model.add(Dense(len(label_encoder.classes_), activation='softmax'))

  # Compile the model
  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  # Train the model
  model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
  from keras.models import load_model

  model.save('my_model.h5')
  model = load_model('my_model.h5')

    # Reshape the test data
  num_samples = X_test.shape[0]
  num_timesteps = X_test.shape[1]
  num_features = X_test.shape[2]
  X_test_reshaped = X_test.reshape(num_samples, num_timesteps, num_features)


  # Use the trained model to make predictions on the test data
  predictions = model.predict(X_test_reshaped)

  # Get the predicted labels
  predicted_labels = np.argmax(predictions, axis=1)

  # Decode the predicted labels using the label_encoder
  predicted_categories = label_encoder.inverse_transform(predicted_labels)

  return predicted_categories

# Create your views here.
@csrf_exempt


def index(request):
    if request.method == 'GET':
        return render(request, 'ad/index.html')
    elif request.method == 'POST':
        try:
            url = request.POST['urlInput']  # Get the URL from the form POST request
            result = main(url)  # Ensure that 'main' returns a list

            # Pass 'result' to the template for rendering
            return render(request, 'ad/index.html', {'pred': result})

        except MultiValueDictKeyError:
            # Handle the case where 'urlInput' is not found in the POST request
            return JsonResponse({'error': 'Missing URL input'})

