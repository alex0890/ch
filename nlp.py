import json
import random
import nltk
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import random_responses  # Import random_responses module

nltk.download('punkt')

def load_json(file):
    """Load a JSON file and return its contents."""
    with open(file) as bot_responses:
        print(f"Loaded '{file}' successfully!")
        return json.load(bot_responses)


response_data = load_json("bot.json")


def tokenize_input(input_string):
    """Tokenize an input string and convert it to lowercase."""
    return nltk.word_tokenize(input_string.lower())


def correct_spelling(input_text):
    """Correct spelling in the input text using TextBlob."""
    corrected_text = TextBlob(input_text).correct()
    return str(corrected_text)


def find_best_match(input_text, possible_responses):
    """Find the best match for the input text from a list of possible responses."""
    vectorizer = TfidfVectorizer(tokenizer=tokenize_input)
    response_vectors = vectorizer.fit_transform(possible_responses + [input_text])
    input_vector = response_vectors[-1]
    similarities = cosine_similarity(response_vectors[:-1], input_vector)
    best_match_index = np.argmax(similarities)
    return possible_responses[best_match_index]


def is_greeting(input_text):
    """Check if the input text contains a greeting."""
    greetings = ['hello', 'hi', 'hey', 'sup']
    return any(greeting in input_text.lower() for greeting in greetings)


def is_farewell(input_text):
    """Check if the input text contains a farewell."""
    farewells = ['goodbye', 'bye', 'see you', 'later']
    return any(farewell in input_text.lower() for farewell in farewells)


def get_response(input_string):
    """Generate an appropriate response for the given input string."""
    if not input_string:
        return "Please type something so we can chat :("

    input_string = input_string.lower()

    if is_greeting(input_string):
        return random.choice(random_responses.greetings)  # Use random_responses for greetings

    if is_farewell(input_string):
        return random.choice(random_responses.farewells)  # Use random_responses for farewells

    try:
        corrected_input = correct_spelling(input_string)
        tokenized_input = tokenize_input(corrected_input)

        possible_responses = [response["patterns"] for response in response_data]
        flattened_responses = [item for sublist in possible_responses for item in sublist]
        best_matching_response = find_best_match(" ".join(tokenized_input), flattened_responses)
        response_index = [index for index, sublist in enumerate(possible_responses) if best_matching_response in sublist][0]

        response = random.choice(response_data[response_index]["responses"])

        return response
    except ValueError as e:
        print(f"Error: {e}")
        return "I'm sorry, I couldn't understand your question."

def update_bot_json(tag, pattern, response):
    new_entry = {
        "tag": tag,
        "patterns": [pattern],
        "responses": [response]
    }

    response_data.append(new_entry)
    with open("bot.json", "w") as bot_responses:
        json.dump(response_data, bot_responses, indent=2)
        print("bot.json updated successfully!")

def load_response_data(filename):
    global response_data
    with open(filename, "r") as bot_responses:
        response_data = json.load(bot_responses)
