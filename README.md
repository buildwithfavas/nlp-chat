# Intelligent Retrieval-Based Chatbot

A simple yet powerful AI chatbot built using Python, Flask, and NLTK. This project uses TF-IDF and Cosine Similarity to understand user queries and retrieve the most relevant answers from a knowledge corpus.

## Features

- **AI-Powered**: Uses Natural Language Processing (NLP) to understand context.
- **Web Interface**: A modern, responsive chat UI built with HTML/CSS/JS.
- **Customizable**: Easily trainable by simply updating the text corpus.
- **Lightweight**: Runs locally without needing external API keys.

## Tech Stack

- **Backend**: Python, Flask
- **NLP**: NLTK (Natural Language Toolkit), Scikit-learn (TF-IDF)
- **Frontend**: HTML5, CSS3, JavaScript (jQuery)

## Installation

1.  **Clone the repository**:

    ```bash
    git clone <your-repo-url>
    cd Live-Chatbot-for-Final-Year-Project
    ```

2.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**:

    ```bash
    python app.py
    ```

4.  **Open in Browser**:
    Go to [http://localhost:5000](http://localhost:5000)

## How it Works

The chatbot uses a **Retrieval-Based** model:

1.  **Corpus**: The bot reads a text dataset (`chatbot.py`).
2.  **Vectorization**: It converts user input and corpus sentences into TF-IDF vectors.
3.  **Similarity**: It calculates Cosine Similarity to find the closest matching sentence.
4.  **Response**: The best match is returned to the user.

## Project Structure

- `app.py`: The Flask web server.
- `chatbot.py`: The core NLP logic and text corpus.
- `templates/index.html`: The frontend chat interface.
- `requirements.txt`: Python dependencies.

## License

Open Source.
