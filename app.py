from flask import Flask, render_template, request
from chatbot import ChatBot, corpus_text

app = Flask(__name__)

# Initialize the chatbot once when the app starts
bot = ChatBot(corpus_text)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    if not userText:
        return "Please say something!"
    
    # Check for greeting manually since the bot class prints directly in some cases
    # We need to adapt the bot's logic slightly or handle it here.
    # The current bot implementation prints greetings. Let's reuse its logic but capture the return.
    
    greeting = bot.greeting(userText)
    if greeting:
        return greeting
        
    return str(bot.response(userText))

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)
