import codecs
import logging
import os.path
import re
import time

from .lemma.const import LemmatizationStrategy
from .text_preprocessing import TextPreprocessor, StopWordsRemover


logger = logging.getLogger(__name__)


class LineIterator:
    """
    Iterator for the lines of a file.
    Read every line, and process it to produce a list of normalized, filtered words.
    """

    def __init__(self, path, text_preprocessor, remove_stop_words=True, include_words_regexs=None,
                 exclude_words_regexs=None,
                 lemmatize=False, lemmatization_strategy=LemmatizationStrategy.ALL):
        """
        :param path: Path of the directory to explore.
        :param text_preprocessor: Instance of altran.text_preprocessing.TextPreprocessor.TextPreprocessor class
        :param remove_stop_words: Whether to remove stop words
        :param inc: List of regular expressions matching the files to be included into
            the processing. If left None, all files will be included.
        :param exc: List of regular expressions matching the files to be excluded into
            the processing among those considered as included in the inc parameter. If left
            None, no file will be excluded.
        :param include_words_regexs: List of regular expressions (strings) that must be matched
            by any document word.
        :param exclude_words_regexs: List of regular expressions (strings) that ust not be
            matched by any document word.
        :param lemmatize: Whether lemmatize the text. Default, False.
        :param lemmatization_strategy: Strategy to use in lemmatization (see .lemma.const.py)
        """
        self.path = path
        self.remove_stop_words = remove_stop_words
        self.include_words_regexs = include_words_regexs
        self.exclude_words_regexs = exclude_words_regexs
        self.preprocessor = text_preprocessor
        self.lemmatize = lemmatize
        self.lemmatization_strategy = lemmatization_strategy

    def __iter__(self):
        with codecs.open(self.path, "r", "utf-8") as f:
            for l in f:
                if len(l) > 0:
                    yield self.preprocessor.preprocessText(l, remove_stop_words=self.remove_stop_words,
                                                           include_words_regexs=self.include_words_regexs,
                                                           exclude_words_regexs=self.exclude_words_regexs,
                                                           lemmatize=self.lemmatize,
                                                           lemmatization_strategy=self.lemmatization_strategy)


class DocTreeIterator:
    """
    This class iterates along the files located in a directory (including all its subdirectories) and processes the
    text contained in these files. Text processing include:
      1. Normalize
      2. Tokenize
      3. Remove stop words
    As a result, a vector containing normalizaed words is returned on every iteration.
    """

    def __init__(self, path, text_preprocessor, remove_stop_words=False, inc=None, exc=None,
                 include_words_regexs=None,
                 exclude_words_regexs=None, lemmatize=False, lemmatization_strategy=LemmatizationStrategy.ALL):
        """

        :param path: Path of the directory to explore.
        :param text_preprocessor: Instance of altran.text_preprocessing.TextPreprocessor.TextPreprocessor class
        :param remove_stop_words: Whether to remove stop words
        :param inc: List of regular expressions matching the files to be included into
                the processing. If left None, all files will be included.
        :param exc: List of regular expressions matching the files to be excluded into
                the processing among those cosidered as included in the inc parameter. If left
                None, no file will be excluded.
        :param include_words_regexs: List of regular expressions (strings) that must be matched
                by any document word.
        :param exclude_words_regexs: List of regular expressions (strings) that must not be
                matched by any document word.
        :param lemmatize: Whether lemmatize the text. Default, False.
        :param lemmatization_strategy: Strategy to use in lemmatization (see altran.nlp.lemma.const.py)
        """
        self.path = path
        self.preprocessor = text_preprocessor
        self.remove_stop_words = remove_stop_words

        self.lemmatize = lemmatize
        self.lemmatization_strategy = lemmatization_strategy

        self.inc = inc
        self.exc = exc
        self.inc_reg_exps = []
        self.exc_reg_exps = []

        if inc:
            for e in inc:
                self.inc_reg_exps.append(re.compile(e))
        if exc:
            for e in exc:
                self.exc_reg_exps.append(re.compile(e))

        self.include_words_regexs = include_words_regexs
        self.exclude_words_regexs = exclude_words_regexs

        self.paths = self._get_paths(path)

    def __iter__(self):
        # Read files, normalize and tokenize.
        self.counter = 0
        for f in self.paths:
            self.counter += 1
            yield self._process_text_file(f)

    def _visit_dir_add_to_vector(self, arg, dirname, names):
        """
        Procedure to be executed by a directory visitor.
        See https://docs.python.org/2/library/os.path.html
        """
        for name in names:
            path = os.path.join(dirname, name)
            if os.path.isfile(path):
                if self._matches(name):
                    arg.append(path)
                else:
                    logging.log(logging.INFO, 'File excluded: ' + path)
            elif os.path.islink(path) and os.path.isdir(path):
                # Manage symbolic links to directories
                os.path.walk(path, self._visit_dir_add_to_vector, arg)

    def _matches(self, s):
        if len(self.inc_reg_exps) > 0:
            # Check whether file should not be included
            not_included = True
            for r in self.inc_reg_exps:
                if (r.match(s) != None):
                    not_included = False
                    break
            if not_included:
                return False

        if len(self.exc_reg_exps) > 0:
            # Check wether file should be excluded
            for r in self.exc_reg_exps:
                if (r.match(s) != None):
                    return False
        return True

    def _get_paths(self, directory):
        """
        Get the paths of the all the files contained in a given directory.

        Args:
            directory: The folder to explore.
        """
        paths = []
        os.path.walk(directory, self._visit_dir_add_to_vector, paths)
        return paths

    def _process_text_file(self, path):
        logging.log(logging.DEBUG, 'Processing file ' + str(self.counter) + '/' + str(len(self.paths)) + ': ' + path)
        start_time = time.time()

        t1 = time.time()
        f = open(path)
        text = '\n'.join(f.readlines())
        f.close()
        t2 = time.time()
        logging.log(logging.DEBUG, 'Read file in ' + str((t2 - t1) * 1000) + ' ms')

        t1 = time.time()
        res = self.preprocessor.preprocessText(text,
                                               remove_stop_words=self.remove_stop_words,
                                               include_words_regexs=self.include_words_regexs,
                                               exclude_words_regexs=self.exclude_words_regexs,
                                               lemmatize=self.lemmatize,
                                               lemmatization_strategy=self.lemmatization_strategy)

        t2 = time.time()
        logging.log(logging.DEBUG, 'Preprocessed text in ' + str((t2 - t1) * 1000) + ' ms')

        t1 = time.time()
        res = self.preprocessor.removeEmptyStrings(res)
        t2 = time.time()
        logging.log(logging.DEBUG, 'Remove empty strings in ' + str((t2 - t1) * 1000) + ' ms')

        end_time = time.time()
        logging.log(logging.DEBUG,
                    'Preprocessed file ' + str(self.counter) + '/' + str(len(self.paths)) + ': ' + path + ' in ' + str(
                        (end_time - start_time) * 1000) + ' milliseconds')

        return res
