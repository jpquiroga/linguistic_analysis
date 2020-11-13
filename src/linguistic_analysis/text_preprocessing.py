import logging
import nltk
from nltk.corpus import stopwords
import re
from typing import Iterable, List, Text

from .lemma.const import LemmatizationStrategy

logger = logging.getLogger(__name__)


class TextPreprocessor(object):
    """
    General text preprocessor.
    This class makes possible to convert any text to a normalized text or to a vector of normalized words.
    """

    def __init__(self, nltk_data_dir: str, languages: Iterable[Text]):
        """
        :param nltk_data_dir: Directory containing the `nltk` corpora data for stop words.
        :param languages: List of languages to be processed (['spanish', 'english']).
        """
        # Encoding compatible with python 3
        self.TWEET_NORMALIZE_REGEXP = re.compile('[\\.,;:¡!?¿\\s\\(\\)\\[\\]\\{\\}\\-\\\'\\"\\\\\\|\xa0]')
        self.TEXT_NORMALIZE_REGEXP = re.compile('[\\.,;:!¡?¿\\s\\(\\)\\[\\]\\{\\}\\-\\\'\\"\\\\#\\|\xa0]')
        self.URL_REGEXP = re.compile('(https?|ftp)://[^\\s]+')
        self.EMAIL_REGEXP = re.compile('[^\\s]+@[^\\s]+')
        self.REMOVE_ACCENTS_A_REGEXP = re.compile('[á|à|ä|â|ã]')
        self.REMOVE_ACCENTS_E_REGEXP = re.compile('[é|è|ë|ê]')
        self.REMOVE_ACCENTS_I_REGEXP = re.compile('[í|ì|ï|î]')
        self.REMOVE_ACCENTS_O_REGEXP = re.compile('[ó|ò|ö|ô|õ]')
        self.REMOVE_ACCENTS_U_REGEXP = re.compile('[ú|ù|ü|û]')

        self.remove_accents_translate_table = {}
        self.remove_accents_translate_table.update(str.maketrans("áàäâã", "aaaaa"))
        self.remove_accents_translate_table.update(str.maketrans("éèëê", "eeee"))
        self.remove_accents_translate_table.update(str.maketrans("íìïî", "iiii"))
        self.remove_accents_translate_table.update(str.maketrans("óòöôõ", "ooooo"))
        self.remove_accents_translate_table.update(str.maketrans("úùüû", "uuuu"))
        self.remove_accents_translate_table.update(str.maketrans("ÁÀÄÂ", "AAAA"))
        self.remove_accents_translate_table.update(str.maketrans("ÉÈËÊ", "EEEE"))
        self.remove_accents_translate_table.update(str.maketrans("ÍÌÏÎ", "IIII"))
        self.remove_accents_translate_table.update(str.maketrans("ÓÒÖÔ", "OOOO"))
        self.remove_accents_translate_table.update(str.maketrans("ÚÙÜÛ", "UUUU"))

        self.nltk_data_dir = nltk_data_dir
        nltk.data.path[0] = nltk_data_dir
        self.languages = languages
        self.stop_words_remover = StopWordsRemover(languages)

        # Lemmatizer table (one per language)
        self.lemmatizers = {_l: None for _l in languages}

#    def text_to_str(self, s):
#        if isinstance(s, str):
#            return s
#        elif isinstance(s, unicode):
#            return s.encode('utf-8', 'ignore')

#    def text_to_unicode(self, s):
#        if isinstance(s, str):
#            return s.decode('utf-8', 'ignore')
#        elif isinstance(s, unicode):
#            return s

    def normalize(self, s: Text) -> Text:
        """
        Normalize a text string.

        :param s: The text to be normalized.

        :return: The text normalized.
        """
        res = s.lower()
        pattern = self.TEXT_NORMALIZE_REGEXP

        logger.debug("Text with URL and e-mail: " + res)
        res = self.URL_REGEXP.sub(' ', res)
        res = self.EMAIL_REGEXP.sub(' ', res)
        logger.debug("Text without URL and e-mail but with punctuation chars: " + res)
        res = pattern.sub(' ', res)
        logger.debug("Text without punctuation chars: " + res)
        logger.debug("Text Normalized (with accents): " + res)
        res = self.remove_accents(res)
        logger.debug("Text Normalized (without accents): " + res)

        return res

    def remove_accents(self, s):
        """
        Remove accents from vowels.

        Args:
            s: The text that will have the accents removed.

        Returns:
            The text with accents from vowels removed.
        """
        return s.translate(self.remove_accents_translate_table)

    def remove_stop_words(self, words: Iterable[Text]) -> List[Text]:
        """
        Remove the stop words from an array of words.

        :param words: The tokenized text to process.
        :return: List of words without stop words.
        """
        return self.stop_words_remover.remove_stop_words(words)

    def split_by_regex(self, s: Text, regex: Text) -> List[Text]:
        """
        Split a string with respect to a regular expression.

        :param s: String to be split.
        :param regex: Regular expression identifying the splitting pattern.

        :return: List of identified tokens.
        """
        p = re.compile(regex)
        return p.split(s)

    def remove_empty_strings(self, string_vector: Iterable[Text]) -> List[Text]:
        return [w for w in string_vector if len(w) > 0]

    def remove_words_regex(self, words: Iterable[Text], regexps: Iterable[Text], match: bool) -> List[Text]:
        """
        Remove words from a vector matching a given regular expression

        :param words: Vector of words to be processed.
        :param regex: Vector of strings containing regular expressions to be matched.
        :param match: If True, any matching word will be removed. If False,
                      any non matching words will be deleted.

        :return: A vector without the removed words.
        """
        compiled_regs = [re.compile(r) for r in regexps]
        return [w for w in words if self._match_any(w, compiled_regs) != match]

    def _match_any(self, word: Text, compiled_regex_list: Iterable[object]) -> bool:
        """
        :param word: Word to be checked
        :param compiled_regex_list: List (or vector) of compiled regular expressions (with re.compile()) to be matched.

        :return: True if word matches any regular expression in compiled_regex_list
        """
        for reg in compiled_regex_list:
            if reg.match(word):
                return True
        return False

    def preprocess_text(self, text: Text, remove_stop_words: bool = False, include_words_regexs: Iterable[object] = None,
                       exclude_words_regexs: Iterable[object] = None, lemmatize: bool = False,
                       lemmatization_strategy: Text = LemmatizationStrategy.ALL) -> List[Text]:
        """
        Preprocess a text, returning a vector of normalized tokens (words). The following flow is
        executed: normalize -> tokenize -> remove empty words -> remove stop words -> filter words

        :param text: The text to process.
        :param remove_stop_words: Whether to remove stop words for current languages.
        :param include_words_regexs: List of regular expressions that must match words to be returned in resulting vector.
            If a word matches any of the regular expression it will be included in the resulting vector.
            If None or empty, no filtering is made and all words are included.

        :param exclude_words_regexs: List of regular expressions that must match words to be removed from the resulting vector.
              If a word matches any of the regular expression it will be removed from the resulting vector.
              If None or empty, no filtering is made and no word is removed.
              **IMPORTANT**: This filter is applied after include_words_regex.
        :param lemmatize: Whether lemmatize the text. Default, False.
        :param lemmatization_strategy: Strategy to use in lemmatization (see altran.nlp.lemma.const.py)

        :return: A vector of tokens (words).
        """
        logging.log(logging.DEBUG, 'Start preprocessText')
        logging.log(logging.DEBUG, "Original Text: " + text)
        logging.log(logging.DEBUG, "Text converted to str: " + text)

        res = self.split_by_regex(self.normalize(text), '\\s')

        logging.log(logging.DEBUG, "Text Normalized and split by Regex (\\\\s): " + str(res))
        res = self.remove_empty_strings(res)
        if remove_stop_words:
            logging.debug(logging.DEBUG, 'Start removeStopWords')
            res = self.remove_stop_words(res)
            logging.log(logging.DEBUG, 'End removeStopWords')
        logging.log(logging.DEBUG, "Text with stop words removed: " + str(res))
        if include_words_regexs:
            logging.log(logging.DEBUG, 'Start (include_words) removeWordsRegex')
            res = self.remove_words_regex(res, include_words_regexs, False)
            logging.log(logging.DEBUG, 'End (include_words) removeWordsRegex')
        logging.log(logging.DEBUG, "Text with words which don't match include_words_regexs removed: " + str(res))
        if exclude_words_regexs:
            logging.log(logging.DEBUG, 'Start (exclude_words) removeWordsRegex')
            res = self.remove_words_regex(res, exclude_words_regexs, True)
            logging.log(logging.DEBUG, 'End (exclude_words) removeWordsRegex')
        logging.log(logging.DEBUG, "Text with words which match exclude_words_regexs removed: " + str(res))

        if lemmatize:
            logger.error("Lemmatization is not yet supported!")
#            logging.debug("Start lemmatization")
#            # Use the lemmatizer of the first language
#            _main_language = self.languages[0]
#            if _main_language == "spanish":
#                if self.lemmatizers[_main_language] == None:
#                    self.lemmatizers[_main_language] = SpanishLemmatizer(self)
#                _lemmatizer = self.lemmatizers[_main_language]
#                res = _lemmatizer.lemmatize_text(res, lemmatization_strategy)
#
#            # TODO Add new languages
#
#            logging.debug("End lemmatization")

        logger.debug('End preprocess_text')
        return res


class StopWordsRemover(object):
    """
    Remove stop words in an optimized way.
    """

    def __init__(self, languages: Iterable[Text]):
        """
        :param languages: List of languages to remove stop words (['spanish', 'english'])
        """
        self.languages = languages
        self.stop_words = {}
        for lang in languages:
            d = {}
            self.stop_words[lang] = d
            for w in stopwords.words(lang):
                d[w] = w

    def remove_stop_words(self, words: Iterable[Text]) -> List[Text]:
        """
        Remove the stop words from an array of words.

        :param words: The tokenized text to process.

        :return: A list of words without stop words.
        """
        res = words
        for lang in self.languages:
            res = [w for w in res if not w in self.stop_words[lang]]
        return res
