from __future__ import unicode_literals
from .logic_adapter import LogicAdapter


class Sentiment(LogicAdapter):
    """
    A logic adapter that returns a response based on known responses to
    the closest matches to the sentiment of the input statement.
    """

    def get(self, input_statement, dm, vec_model):
        """
        Takes a statement string and a list of statement strings.
        Returns the closest matching statement from the list.
        """
        statement_list = self.chatbot.storage.get_response_statements()

        if not statement_list:
            if self.chatbot.storage.count():
                # Use a randomly picked statement
                self.logger.info(
                    'No statements have known responses. ' +
                    'Choosing a random response to return.'
                )
                random_response = self.chatbot.storage.get_random()
                random_response.confidence = 0
                return random_response
            else:
                raise self.EmptyDatasetException()

        closest_match = input_statement
        closest_match.confidence = 0

        # Find the closest matching known statement
        for statement in statement_list:
            confidence = self.compare_statements(input_statement, statement, dm, vec_model)

            if confidence > closest_match.confidence:
                statement.confidence = confidence
                closest_match = statement

        return closest_match

    def can_process(self, statement):
        """
        Check that the chatbot's storage adapter is available to the logic
        adapter and there is at least one statement in the database.
        """
        return self.chatbot.storage.count()

    def process(self, input_statement, dm, vec_model):

        # Select the closest match to the input statement
        closest_match = self.get(input_statement, dm, vec_model)
        self.logger.info('Using "{}" as a close match to "{}"'.format(
            input_statement.text, closest_match.text
        ))

        # Print the closest match score
        wordsInClosestMatch = nltk.tokenize.casual_tokenize(closest_match.text, preserve_case=False)

        numWordsFromClosestMatchInLexicon = 0
        closestMatchMood = 0

        for word in wordsInClosestMatch:

            #skip punctuation and urls... could do this with the tokenizer if I had more time to read the docs!
            if len(word) > 1 or (word == 'i' or word == 'a'):

                #[6] is happy, [8] is sad
                if word in mood_lexicon:
                    closestMatchMood = closestMatchMood + mood_lexicon[word][6]
                    numWordsFromClosestMatchInLexicon = numWordsFromClosestMatchInLexicon + 1

                if (numWordsFromClosestMatchInLexicon > 0):
                    closestMatchMood = closestMatchMood / numWordsFromOtherStatementInLexicon

        print("OUTSCORE: " + closestMatchMood)

        # Get all statements that are in response to the closest match
        response_list = self.chatbot.storage.filter(
            in_response_to__contains=closest_match.text
        )

        if response_list:
            self.logger.info(
                'Selecting response from {} optimal responses.'.format(
                    len(response_list)
                )
            )
            response = self.select_response(input_statement, response_list)
            response.confidence = closest_match.confidence
            self.logger.info('Response selected. Using "{}"'.format(response.text))
        else:
            response = self.chatbot.storage.get_random()
            self.logger.info(
                'No response to "{}" found. Selecting a random response.'.format(
                    closest_match.text
                )
            )

            # Set confidence to zero because a random response is selected
            response.confidence = 0

        return response
