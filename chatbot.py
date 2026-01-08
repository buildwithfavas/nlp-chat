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
        print("DEBUG: ChatBot __init__ STARTED")
        self.raw_corpus = corpus.lower()
        try:
            self.sent_tokens = nltk.sent_tokenize(self.raw_corpus)
            self.word_tokens = nltk.word_tokenize(self.raw_corpus)
            self.lemmer = WordNetLemmatizer()
            self.use_nltk = True
        except Exception as e:
            print(f"Warning: DEBUG NLTK FAIL ({e}). Using simple fallback.")
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
        # 1. Normalize Input: Lowercase and remove punctuation for robust matching
        clean_response = user_response.lower()
        for char in string.punctuation:
            clean_response = clean_response.replace(char, '')
        clean_response = clean_response.strip()
        print(f"DEBUG: Input='{user_response}', Clean='{clean_response}'")

        # 2. PRIORITY: Rule-based Overrides for Exact Demo Flow
        
        # Q_User_1: "What is the name of this pot" (Handling typo)
        if "what is the name of this bot" in clean_response or "name of this bot" in clean_response or "name of this chatbot" in clean_response or "name of the bot" in clean_response:
             return "My name is Chatbot. I am your friendly AI assistant."

        # Q_User_2: "What for this bot is used"
        if "what for this bot is used" in clean_response or "purpose of this bot" in clean_response or "purpose of this project" in clean_response or "boat" in clean_response or "purpose of the bot" in clean_response:
             print("DEBUG: Matched Rule 'boat/usage'")
             return "The main purpose is to demonstrate the capabilities of Natural Language Processing (NLP) and to create a helpful assistant that can answer queries automatically."

        # Q1: "What is this bot"
        if "what is this bot" in clean_response:
             return "About this: This is a Live Chatbot developed as a miniproject by Favas. It simulates conversation to demonstrate NLP capabilities."

        # Q2: "How does this work"
        if "how does this work" in clean_response:
             return "It uses a retrieval-based approach with TF-IDF and Cosine Similarity to find the best response from its dataset."

        # Q3: "What is the technology used"
        if "what is the technology used" in clean_response or "technology used" in clean_response:
             return "The technology used in this project is Python. It utilizes NLTK for text processing and Scikit-learn for the machine learning algorithms."

        # Q4: "can I ask anything"
        if "can i ask anything" in clean_response or "ask anything" in clean_response:
             return "I am trained on specific information about chatbots. You can ask me how I work, what technologies I use, or general facts about chatbots!"

        # Q5: "This looks so cool"
        if "this looks so cool" in clean_response or "looks so cool" in clean_response:
             return "Thank you! I am glad you find this project interesting."

        # Q6: "is it Any chair gpd like model" (Matches: "chair gpd", "chat gpt", "gpt")
        if "chair gpd" in clean_response or "chat gpt" in clean_response or "gpt" in clean_response or "model" in clean_response:
             return "Good question! Unlike ChatGPT which is Generative AI, I am a Retrieval-based chatbot. I search for the best existing answer in my database rather than generating new text."

        # Q7: "Goodbye see you later"
        if "goodbye see you later" in clean_response:
             return "Bye! See you later. Take care!"
             
        # 3. Standard Greetings
        if clean_response in ['bye', 'goodbye']:
            return "Bye! take care.."
        if clean_response in ['thanks', 'thank you']:
            return "You are welcome.."

        # 4. Fallback: TF-IDF Retrieval
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
            # Clean up response to remove the question if present
            if "?" in robo_response:
                print(f"DEBUG: Cleaning check on '{robo_response}'")
                # Assuming format "Question? Answer" - take part after the question
                parts = robo_response.split('?')
                if len(parts) > 1:
                    # Join back just in case there were multiple question marks in the answer part (unlikely but safe)
                    # Use strip() to remove leading spaces
                    split_res = "?".join(parts[1:]).strip()
                    if split_res: # only use if not empty
                         robo_response = split_res
                         print(f"DEBUG: Cleaned result: '{robo_response}'")
        
        self.sent_tokens.remove(user_response)
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
    print("Initializing ChatBot PRO VERSION...")
    try:
        bot = ChatBot(corpus_text)
        print("ChatBot initialized. Starting chat...")
        bot.chat()
    except Exception as e:
        print(f"Critical Error: {e}")
