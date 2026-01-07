import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK data is downloaded
# (Skipping explicit check to avoid startup crashes - utilizing fallbacks if missing)

class ChatBot:
    def __init__(self, corpus):
        self.raw_corpus = corpus.lower()
        try:
            self.sent_tokens = nltk.sent_tokenize(self.raw_corpus)
            self.word_tokens = nltk.word_tokenize(self.raw_corpus)
            self.lemmer = WordNetLemmatizer()
            self.use_nltk = True
        except Exception as e:
            print(f"Warning: NLTK initialization failed ({e}). Using simple fallback.")
            self.sent_tokens = self.raw_corpus.split('.')
            self.word_tokens = self.raw_corpus.split()
            self.lemmer = None
            self.use_nltk = False

    def lem_tokens(self, tokens):
        if not self.use_nltk or not self.lemmer:
            return tokens
        return [self.lemmer.lemmatize(token) for token in tokens]

    def lem_normalize(self, text):
        remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        clean_text = text.lower().translate(remove_punct_dict)
        if self.use_nltk:
            return self.lem_tokens(nltk.word_tokenize(clean_text))
        else:
            return clean_text.split()

    def greeting(self, sentence):
        greeting_inputs = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
        greeting_responses = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
        
        for word in sentence.split():
            if word.lower() in greeting_inputs:
                return random.choice(greeting_responses)
        return None

    def response(self, user_response):
        robo_response = ''
        self.sent_tokens.append(user_response)
        
        TfidfVec = TfidfVectorizer(tokenizer=self.lem_normalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(self.sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx = vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        
        if(req_tfidf == 0):
            robo_response = robo_response + "I am sorry! I don't understand you"
        else:
            robo_response = robo_response + self.sent_tokens[idx]
        
        self.sent_tokens.remove(user_response) # cleanup
        return robo_response

    def chat(self):
        print("CHATBOT: My name is Chatbot. I will answer your queries about Chatbots. If you want to exit, type Bye!")
        while True:
            try:
                user_response = input("YOU: ").lower()
            except (EOFError, KeyboardInterrupt):
                print("\nCHATBOT: Input stream closed. Exiting.")
                break
            except Exception as e:
                print(f"\nCHATBOT: Error reading input: {e}")
                break
                
            if(user_response != 'bye'):
                if(user_response == 'thanks' or user_response == 'thank you' ):
                    print("CHATBOT: You are welcome..")
                    break
                else:
                    if(self.greeting(user_response) != None):
                        print("CHATBOT: "+self.greeting(user_response))
                    else:
                        print("CHATBOT: ",end="")
                        print(self.response(user_response))
            else:
                print("CHATBOT: Bye! take care..")
                break

# Sample corpus text about chatbots
corpus_text = """
A chatbot is an artificial intelligence (AI) software that can simulate a conversation (or a chat) with a user in natural language through messaging applications, websites, mobile apps or through the telephone.
Why are chatbots important? A chatbot is often described as one of the most advanced and promising expressions of interaction between humans and machines. However, from a technological point of view, a chatbot only represents the natural evolution of a Question Answering system leveraging Natural Language Processing (NLP). Formulating responses to questions in natural language is one of the most typical Examples of Natural Language Processing applied in various enterprises’ end-use applications.
There are two main types of chatbots: Rule-based agents and Artificial Intelligence (AI) - based chatbots.
Rule-based agents use a set of predefined rules to answer questions. They are great for simple queries but fail to handle complex conversations.
AI-based chatbots use machine learning and NLP to understand the context and intent of the user. They learn from previous interactions and improve over time.

About this Project:
What is this project? This is a Live Chatbot developed as miniproject by Favas. It is a software program designed to simulate conversation with human users.
What is the purpose of this project? The main purpose is to demonstrate the capabilities of Natural Language Processing (NLP) and to create a helpful assistant that can answer queries automatically.
How does this chatbot work? This chatbot works using a retrieval-based approach. It uses TF-IDF (Term Frequency-Inverse Document Frequency) to vectorise text and Cosine Similarity to find the most similar response from its dataset to the user's input.
What is the technology used? This project is built using the Python programming language. It uses the NLTK (Natural Language Toolkit) library for processing text and the Scikit-learn library for the machine learning algorithms (TF-IDF and Cosine Similarity).
What is the name of this bot? My name is Chatbot. I am your friendly AI assistant.
Why use Python for this? Python is used because it has powerful and easy-to-use libraries like NLTK and Scikit-learn which make implementing Artificial Intelligence and NLP tasks efficient.
"""

if __name__ == "__main__":
    print("Initializing ChatBot...")
    try:
        bot = ChatBot(corpus_text)
        print("ChatBot initialized. Starting chat...")
        bot.chat()
    except Exception as e:
        print(f"Critical Error: {e}")
