from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

chatbot = ChatBot('SentiBot',
                  storage_adapter='chatterbot.storage.SQLStorageAdapter',
                  read_only=True,
                  output_adapter='chatterbot.output.TerminalAdapter',
                  logic_adapters=[
                      {
                          "import_path": "chatterbot.logic.Sentiment",
                          "statement_comparison_function": "chatterbot.comparisons.sentiment_distance",
                          "response_selection_method": "chatterbot.response_selection.get_first_response"
                      }
                    ],
                  database='./db.sqlite3')

chatbot.set_trainer(ChatterBotCorpusTrainer)

chatbot.train(
    "chatterbot.corpus.english"
)

with open('trimmed_twitter_corpus.csv') as f:
    for line in f:
        print(line, end=" ")
        bot_input = chatbot.get_response(line)

