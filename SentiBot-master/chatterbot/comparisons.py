# -*- coding: utf-8 -*-
import sys
import nltk.tokenize
import gensim


"""
This module contains various text-comparison algorithms
designed to compare one statement to another.
"""

# Use python-Levenshtein if available
try:
    from Levenshtein.StringMatcher import StringMatcher as SequenceMatcher
except ImportError:
    from difflib import SequenceMatcher


class Comparator:

    def __call__(self, statement_a, statement_b):
        return self.compare(statement_a, statement_b)

    def compare(self, statement_a, statement_b):
        return 0

    def get_initialization_functions(self):
        """
        Return all initialization methods for the comparison algorithm.
        Initialization methods must start with 'initialize_' and
        take no parameters.
        """
        initialization_methods = [
            (
                method,
                getattr(self, method),
            ) for method in dir(self) if method.startswith('initialize_')
        ]

        return {
            key: value for (key, value) in initialization_methods
        }

class SentimentDistance(Comparator):
    """
    Compare two statements based on the average distance
    of their sentiment scores, based on the DepecheMood++
    lexicon.

    This calculates the distance in mood for each of the
    seven mood scores, then computes the average distance
    to determine the confidence level.
    """

    def __call__(self, statement_a, statement_b, mood_lexicon, vec_model):
        return self.compare(statement_a, statement_b, mood_lexicon, vec_model)

    def compare(self, statement, other_statement, mood_lexicon, vec_model):
        """
        Compare the two input statements.

        :return: 1 - the average distance in mood between the statements.
        ;rtype: float
        """

        if not statement.text or not other_statement.text:
            return 0

        # split and tokenize each statement into words
        tokenizeInput = nltk.tokenize.casual.TweetTokenizer(preserve_case=False, strip_handles=True)
        wordsInStatement = tokenizeInput.tokenize(statement.text)
        wordsInOtherStatement = nltk.tokenize.casual_tokenize(other_statement.text, preserve_case=False)

        numWordsFromStatementInLexicon = 0
        numWordsFromOtherStatementInLexicon = 0

        statementMood = 0
        otherStatementMood = 0

        # score each word and combine scores, then average
        for word in wordsInStatement:

            #skip punctuation and urls... could do this with the tokenizer if I had more time to read the docs!
            if len(word) > 1 or (word == 'i' or word == 'a'):

                # If a word's not present, use the word2vec corpus to find a suitable alternative
#                if word not in mood_lexicon:
#                    try:
#                        result = vec_model.most_similar(positive = [word], topn = 10)
#                        for similarWord in result:
#                            if similarWord[0] in mood_lexicon:
#                                word = similarWord[0]
#                                break
#                    except KeyError:
#                        break

                #[6] is happy, [8] is sad
                if word in mood_lexicon:
                    statementMood = statementMood + mood_lexicon[word][6]
                    numWordsFromStatementInLexicon = numWordsFromStatementInLexicon + 1

                if (numWordsFromStatementInLexicon > 0):
                    statementMood = statementMood / numWordsFromStatementInLexicon
                    print("INSCORE: " + statementMood)

        for word in wordsInOtherStatement:

            #skip punctuation and urls... could do this with the tokenizer if I had more time to read the docs!
            if len(word) > 1 or (word == 'i' or word == 'a'):

                #[6] is happy, [8] is sad
                if word in mood_lexicon:
                    otherStatementMood = otherStatementMood + mood_lexicon[word][6]
                    numWordsFromOtherStatementInLexicon = numWordsFromOtherStatementInLexicon + 1

                if (numWordsFromOtherStatementInLexicon > 0):
                    otherStatementMood = otherStatementMood / numWordsFromOtherStatementInLexicon

        # get list of average scores
        distance = abs(otherStatementMood - statementMood)

        # return 1 - average distance
        return 1 - distance

class LevenshteinDistance(Comparator):
    """
    Compare two statements based on the Levenshtein distance
    of each statement's text.

    For example, there is a 65% similarity between the statements
    "where is the post office?" and "looking for the post office"
    based on the Levenshtein distance algorithm.
    """

    def compare(self, statement, other_statement):
        """
        Compare the two input statements.

        :return: The percent of similarity between the text of the statements.
        :rtype: float
        """

        PYTHON = sys.version_info[0]

        # Return 0 if either statement has a falsy text value
        if not statement.text or not other_statement.text:
            return 0

        # Get the lowercase version of both strings
        if PYTHON < 3:
            statement_text = unicode(statement.text.lower()) # NOQA
            other_statement_text = unicode(other_statement.text.lower()) # NOQA
        else:
            statement_text = str(statement.text.lower())
            other_statement_text = str(other_statement.text.lower())

        similarity = SequenceMatcher(
            None,
            statement_text,
            other_statement_text
        )

        # Calculate a decimal percent of the similarity
        percent = round(similarity.ratio(), 2)

        return percent


class SynsetDistance(Comparator):
    """
    Calculate the similarity of two statements.
    This is based on the total maximum synset similarity between each word in each sentence.

    This algorithm uses the `wordnet`_ functionality of `NLTK`_ to determine the similarity
    of two statements based on the path similarity between each token of each statement.
    This is essentially an evaluation of the closeness of synonyms.
    """

    def initialize_nltk_wordnet(self):
        """
        Download required NLTK corpora if they have not already been downloaded.
        """
        from .utils import nltk_download_corpus

        nltk_download_corpus('corpora/wordnet')

    def initialize_nltk_punkt(self):
        """
        Download required NLTK corpora if they have not already been downloaded.
        """
        from .utils import nltk_download_corpus

        nltk_download_corpus('tokenizers/punkt')

    def initialize_nltk_stopwords(self):
        """
        Download required NLTK corpora if they have not already been downloaded.
        """
        from .utils import nltk_download_corpus

        nltk_download_corpus('corpora/stopwords')

    def compare(self, statement, other_statement):
        """
        Compare the two input statements.

        :return: The percent of similarity between the closest synset distance.
        :rtype: float

        .. _wordnet: http://www.nltk.org/howto/wordnet.html
        .. _NLTK: http://www.nltk.org/
        """
        from nltk.corpus import wordnet
        from nltk import word_tokenize
        from chatterbot import utils
        import itertools

        tokens1 = word_tokenize(statement.text.lower())
        tokens2 = word_tokenize(other_statement.text.lower())

        # Remove all stop words from the list of word tokens
        tokens1 = utils.remove_stopwords(tokens1, language='english')
        tokens2 = utils.remove_stopwords(tokens2, language='english')

        # The maximum possible similarity is an exact match
        # Because path_similarity returns a value between 0 and 1,
        # max_possible_similarity is the number of words in the longer
        # of the two input statements.
        max_possible_similarity = max(
            len(statement.text.split()),
            len(other_statement.text.split())
        )

        max_similarity = 0.0

        # Get the highest matching value for each possible combination of words
        for combination in itertools.product(*[tokens1, tokens2]):

            synset1 = wordnet.synsets(combination[0])
            synset2 = wordnet.synsets(combination[1])

            if synset1 and synset2:

                # Get the highest similarity for each combination of synsets
                for synset in itertools.product(*[synset1, synset2]):
                    similarity = synset[0].path_similarity(synset[1])

                    if similarity and (similarity > max_similarity):
                        max_similarity = similarity

        if max_possible_similarity == 0:
            return 0

        return max_similarity / max_possible_similarity


class SentimentComparison(Comparator):
    """
    Calculate the similarity of two statements based on the closeness of
    the sentiment value calculated for each statement.
    """

    def initialize_nltk_vader_lexicon(self):
        """
        Download the NLTK vader lexicon for sentiment analysis
        that is required for this algorithm to run.
        """
        from .utils import nltk_download_corpus

        nltk_download_corpus('sentiment/vader_lexicon')

    def compare(self, statement, other_statement):
        """
        Return the similarity of two statements based on
        their calculated sentiment values.

        :return: The percent of similarity between the sentiment value.
        :rtype: float
        """
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        sentiment_analyzer = SentimentIntensityAnalyzer()
        statement_polarity = sentiment_analyzer.polarity_scores(statement.text.lower())
        statement2_polarity = sentiment_analyzer.polarity_scores(other_statement.text.lower())

        statement_greatest_polarity = 'neu'
        statement_greatest_score = -1
        for polarity in sorted(statement_polarity):
            if statement_polarity[polarity] > statement_greatest_score:
                statement_greatest_polarity = polarity
                statement_greatest_score = statement_polarity[polarity]

        statement2_greatest_polarity = 'neu'
        statement2_greatest_score = -1
        for polarity in sorted(statement2_polarity):
            if statement2_polarity[polarity] > statement2_greatest_score:
                statement2_greatest_polarity = polarity
                statement2_greatest_score = statement2_polarity[polarity]

        # Check if the polarity if of a different type
        if statement_greatest_polarity != statement2_greatest_polarity:
            return 0

        values = [statement_greatest_score, statement2_greatest_score]
        difference = max(values) - min(values)

        return 1.0 - difference


class JaccardSimilarity(Comparator):
    """
    Calculates the similarity of two statements based on the Jaccard index.

    The Jaccard index is composed of a numerator and denominator.
    In the numerator, we count the number of items that are shared between the sets.
    In the denominator, we count the total number of items across both sets.
    Let's say we define sentences to be equivalent if 50% or more of their tokens are equivalent.
    Here are two sample sentences:

        The young cat is hungry.
        The cat is very hungry.

    When we parse these sentences to remove stopwords, we end up with the following two sets:

        {young, cat, hungry}
        {cat, very, hungry}

    In our example above, our intersection is {cat, hungry}, which has count of two.
    The union of the sets is {young, cat, very, hungry}, which has a count of four.
    Therefore, our `Jaccard similarity index`_ is two divided by four, or 50%.
    Given our similarity threshold above, we would consider this to be a match.

    .. _`Jaccard similarity index`: https://en.wikipedia.org/wiki/Jaccard_index
    """

    SIMILARITY_THRESHOLD = 0.5

    def initialize_nltk_wordnet(self):
        """
        Download the NLTK wordnet corpora that is required for this algorithm
        to run only if the corpora has not already been downloaded.
        """
        from .utils import nltk_download_corpus

        nltk_download_corpus('corpora/wordnet')

    def compare(self, statement, other_statement):
        """
        Return the calculated similarity of two
        statements based on the Jaccard index.
        """
        from nltk.corpus import wordnet
        import nltk
        import string

        a = statement.text.lower()
        b = other_statement.text.lower()

        # Get default English stopwords and extend with punctuation
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.extend(string.punctuation)
        stopwords.append('')
        lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

        def get_wordnet_pos(pos_tag):
            if pos_tag[1].startswith('J'):
                return (pos_tag[0], wordnet.ADJ)
            elif pos_tag[1].startswith('V'):
                return (pos_tag[0], wordnet.VERB)
            elif pos_tag[1].startswith('N'):
                return (pos_tag[0], wordnet.NOUN)
            elif pos_tag[1].startswith('R'):
                return (pos_tag[0], wordnet.ADV)
            else:
                return (pos_tag[0], wordnet.NOUN)

        ratio = 0
        pos_a = map(get_wordnet_pos, nltk.pos_tag(nltk.tokenize.word_tokenize(a)))
        pos_b = map(get_wordnet_pos, nltk.pos_tag(nltk.tokenize.word_tokenize(b)))
        lemma_a = [
            lemmatizer.lemmatize(
                token.strip(string.punctuation),
                pos
            ) for token, pos in pos_a if pos == wordnet.NOUN and token.strip(
                string.punctuation
            ) not in stopwords
        ]
        lemma_b = [
            lemmatizer.lemmatize(
                token.strip(string.punctuation),
                pos
            ) for token, pos in pos_b if pos == wordnet.NOUN and token.strip(
                string.punctuation
            ) not in stopwords
        ]

        # Calculate Jaccard similarity
        try:
            numerator = len(set(lemma_a).intersection(lemma_b))
            denominator = float(len(set(lemma_a).union(lemma_b)))
            ratio = numerator / denominator
        except Exception as e:
            print('Error', e)
        return ratio >= self.SIMILARITY_THRESHOLD


# ---------------------------------------- #


levenshtein_distance = LevenshteinDistance()
synset_distance = SynsetDistance()
sentiment_distance = SentimentDistance()
jaccard_similarity = JaccardSimilarity()
