import codecs
import langdetect
import logging
import os
import os.path
import re
import shutil
import spacy
import tempfile
import time
from typing import Text

from .text_preprocessing import Iterable, List, TextPreprocessor, StopWordsRemover
from .lemma.const import LemmatizationStrategy

from nltk.tokenize import sent_tokenize

SPACY_MAX_LENGTH = 10000000

logger = logging.getLogger(__name__)


class LineIterator(object):
    """
    Iterator for the lines of a file.
    Read every line, and process it to produce a list of normalized, filtered words.
    """

    def __init__(self, path: Text,
                 text_preprocessor: TextPreprocessor,
                 remove_stop_words: bool = True,
                 include_words_regexs: bool = None,
                 exclude_words_regexs: bool = None,
                 lemmatize: bool = False,
                 lemmatization_strategy: Text = LemmatizationStrategy.ALL):
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
        :param exclude_words_regexs: List of regular expressions (strings) that ust not be
            matched by any document word.
        :param lemmatize: Whether lemmatize the text. Default, False.
        :param lemmatization_strategy: Strategy to use in lemmatization (see altran.nlp.lemma.const.py)

        :param path:
        :param text_preprocessor:
        :param remove_stop_words:
        :param include_words_regexs:
        :param exclude_words_regexs:
        :param lemmatize:
        :param lemmatization_strategy:
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
                    yield self.preprocessor.preprocess_text(l, remove_stop_words=self.remove_stop_words,
                                                            include_words_regexs=self.include_words_regexs,
                                                            exclude_words_regexs=self.exclude_words_regexs,
                                                            lemmatize=self.lemmatize,
                                                            lemmatization_strategy=self.lemmatization_strategy)


class DocTreeIterator(object):
    """
    This class iterates along the files located in a directory (including all its subdirectories) and processes the
    text contained in these files. Text processing include:
      1. Normalize
      2. Tokenize
      3. Remove stop words
    As a result, a vector containing normalized words is returned on every iteration.
    """

    def __init__(self, path: Text,
                 text_preprocessor: TextPreprocessor,
                 remove_stop_words: bool = True,
                 inc: Iterable[Text] = None,
                 exc: Iterable[Text] = None,
                 include_words_regexs: bool = None,
                 exclude_words_regexs: bool = None,
                 lemmatize: bool = False,
                 lemmatization_strategy: Text = LemmatizationStrategy.ALL):
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
                    logger.info('File excluded: ' + path)
            elif os.path.islink(path) and os.path.isdir(path):
                # Manage symbolic links to directories
                os.path.walk(path, self._visit_dir_add_to_vector, arg)

    def _process_dir_add_to_vector(self, file_list, dirname, filenames):
        """
        This method adds to a list all the files contained in a given directory

        :param file_list: list of file paths to be rcompleted. This list is modified by the method
        :param dirname: current directory name
        :param filenames: list of file names in current directory

        :return: The modified file_list
        """
        for name in filenames:
            path = os.path.join(dirname, name)
            if os.path.isfile(path):
                if self._matches(name):
                    file_list.append(path)
                else:
                    logger.info('File excluded: ' + path)
                    #            elif os.path.islink(path) and os.path.isdir(path):
                    #                # Manage symbolic links to directories
                    #                #                os.path.walk(path, self._visit_dir_add_to_vector, arg)
                    #                os.walk(path, self._visit_dir_add_to_vector, arg)
        return file_list

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
            # Check whether file should be excluded
            for r in self.exc_reg_exps:
                if (r.match(s) != None):
                    return False
        return True

    def _get_paths(self, directory):
        """
        Get the paths of the all the files contained in a given directory.

        :param directory: The folder to explore.
        """
        paths = []

        for _directory, _dirnames, _filenames in os.walk(directory):
            self._process_dir_add_to_vector(paths, _directory, _filenames)

            #        os.path.walk(directory, self._visit_dir_add_to_vector, paths)
        return paths

    def _process_text_file(self, path: Text):
        logger.debug('Processing file {}/{}: {}'.format(self.counter, len(self.paths), path))
        start_time = time.time()

        t1 = time.time()
        f = open(path)
        text = '\n'.join(f.readlines())
        f.close()
        t2 = time.time()
        logger.debug('Read file in {} ms'.format((t2 - t1) * 1000))

        t1 = time.time()
        res = self.preprocessor.preprocess_text(text,
                                                remove_stop_words=self.remove_stop_words,
                                                include_words_regexs=self.include_words_regexs,
                                                exclude_words_regexs=self.exclude_words_regexs,
                                                lemmatize=self.lemmatize,
                                                lemmatization_strategy=self.lemmatization_strategy)

        t2 = time.time()
        logger.debug('Preprocessed text in {} ms'.format((t2 - t1) * 1000))

        t1 = time.time()
        res = self.preprocessor.remove_empty_strings(res)
        t2 = time.time()
        logger.debug('Remove empty strings in {} ms'.format((t2 - t1) * 1000))

        end_time = time.time()
        logger.debug('Preprocessed file {}/{}: {} in {} milliseconds'.format(self.counter,
                                                                             len(self.paths),
                                                                             path,
                                                                             (end_time - start_time) * 1000))
        return res


class SentenceSegmenter(object):
    """
    Language independent sentence segmenter based on spacy.
    """

    def __init__(self, spacy_language_models={}, default_language="en"):
        """
        :param spacy_language_models: Spacy language models as a dictionary.
            If their value is None, models are loaded automatically.
            (e.g.: {"es":<>, "en":None, "pt":<>}
        :param default_language: language to use if language inference fails.
        """
        self.languages = [l for l in spacy_language_models.keys()]
        self.spacy_language_models = {}
        for l in spacy_language_models.keys():
            _lm = spacy_language_models[l]
            if _lm is None:
                self.spacy_language_models[l] = spacy.load(l)
                self.spacy_language_models[l].max_length = SPACY_MAX_LENGTH
            else:
                self.spacy_language_models[l] = spacy_language_models[l]

        self.default_language = default_language

    def segment(self, txt: Text, language: Text = None) -> Iterable[Text]:
        """
        :param txt:
        :param language:

        :return: An iterator to the sentences.
        """
        _txt = txt.strip()
        if len(_txt) == 0:
            return None

        if language is None:
            _lan = langdetect.detect(txt)
        else:
            _lan = language

        if _lan not in self.languages:
            _lan = self.default_language
        _nlp = self.spacy_language_models[_lan]
        _iter = _nlp(txt).sents
        return (_s.text for _s in _iter)


class SentenceSegmenterNLTK(object):
    """
    Language independent sentence segmenter based on NLTK.
    """

    def __init__(self):
        pass

    def segment(self, txt: Text, language: Text = None) -> Iterable[Text]:
        """
        :param txt:
        :param language:

        :return: An iterator to the sentences.
        """
        _txt = txt.strip()
        if len(_txt) == 0:
            return None

        return sent_tokenize(_txt)


class SentenceIterator(object):
    """
    Iterator for the sentences of a file.
    Read every line, and process it to produce a list of normalized, filtered words.
    Sentence segmentation is made using spacy.
    """

    def __init__(self, path: Text,
                 text_preprocessor: TextPreprocessor,
                 sentence_segmenter: SentenceSegmenter,
                 remove_stop_words: bool = True,
                 include_words_regexs: Iterable[Text] = None,
                 exclude_words_regexs: Iterable[Text] = None,
                 lemmatize: bool = False,
                 lemmatization_strategy: Text = LemmatizationStrategy.ALL):
        """
        :param path: Path of the directory to explore.
        :param text_preprocessor: Instance of altran.text_preprocessing.TextPreprocessor.TextPreprocessor class
        :param sentence_segmenter:
        :param remove_stop_words: Whether to remove stop words
        :param inc: List of regular expressions matching the files to be included into
            the processing. If left None, all files will be included.
        :param exc: List of regular expressions matching the files to be excluded into
            the processing among those cosidered as included in the inc parameter. If left
            None, no file will be excluded.
        :param include_words_regexs: List of regular expressions (strings) that must be matched
            by any document word.
        :param exclude_words_regexs: List of regular expressions (strings) that ust not be
            matched by any document word.
        :param lemmatize: Whether lemmatize the text. Default, False.
        :param lemmatization_strategy: Strategy to use in lemmatization (see altran.nlp.lemma.const.py)
        """
        self.path = path
        self.remove_stop_words = remove_stop_words
        self.include_words_regexs = include_words_regexs
        self.exclude_words_regexs = exclude_words_regexs
        self.preprocessor = text_preprocessor
        self.sentence_segmenter = sentence_segmenter
        self.lemmatize = lemmatize
        self.lemmatization_strategy = lemmatization_strategy

        with codecs.open(self.path, "r", "utf-8") as f:
            _buffer = ""
            for l in f:
                _buffer += l
            _buffer = self.preprocessor.preprocess_text(_buffer,
                                                        remove_stop_words=self.remove_stop_words,
                                                        include_words_regexs=self.include_words_regexs,
                                                        exclude_words_regexs=self.exclude_words_regexs,
                                                        lemmatize=self.lemmatize,
                                                        lemmatization_strategy=self.lemmatization_strategy)
        self.seg = self.sentence_segmenter.segment(_buffer)

    def __iter__(self):
        return self.seg


class SentenceDocTreeIterator(object):
    """
    This class iterates along the files located in a directory (including all its subdirectories) and processes the
    text contained in these files. Text processing includes:
      0. Sentence segmentation
      1. Normalize
      2. Tokenize
      3. Remove stop words
    As a result, a vector containing normalizaed words is returned on every iteration.
    """

    def __init__(self,
                 path: Text,
                 text_preprocessor: TextPreprocessor,
                 sentence_segmenter: SentenceSegmenter,
                 remove_stop_words: bool = True,
                 inc: Iterable[Text] = None,
                 exc: Iterable[Text] = None,
                 include_words_regexs: Iterable[Text] = None,
                 exclude_words_regexs: Iterable[Text] = None,
                 lemmatize: bool = False,
                 lemmatization_strategy: Text = LemmatizationStrategy.ALL,
                 encoding: Text = None):
        """
        :param path: Path of the directory to explore.
        :param text_preprocessor: Instance of altran.text_preprocessing.TextPreprocessor.TextPreprocessor class
        :param sentence_segmenter:
        :param remove_stop_words: Whether to remove stop words
        :param inc: List of regular expressions matching the files to be included into
            the processing. If left None, all files will be included.
        :param exc: List of regular expressions matching the files to be excluded into
            the processing among those considered as included in the inc parameter. If left
            None, no file will be excluded.
        :param include_words_regexs: List of regular expressions (strings) that must be matched
            by any document word.
        :param exclude_words_regexs: List of regular expressions (strings) that must not be
            matched by any document word.
        :param lemmatize: Whether lemmatize the text. Default, False.
        :param lemmatization_strategy: Strategy to use in lemmatization (see altran.nlp.lemma.const.py)
        :param encoding:
        """
        self.path = path
        self.preprocessor = text_preprocessor
        self.sentence_segmenter = sentence_segmenter
        self.remove_stop_words = remove_stop_words

        self.lemmatize = lemmatize
        self.lemmatization_strategy = lemmatization_strategy

        self.inc = inc
        self.exc = exc
        self.inc_reg_exps = []
        self.exc_reg_exps = []

        self.encoding = encoding

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
            _txt = self._read_file_text(f)
            _sents = self.sentence_segmenter.segment(_txt)
            if _sents is not None:
                for _sent in (_s for _s in _sents):
                    yield self.preprocessor.preprocess_text(_sent,
                                                            remove_stop_words=self.remove_stop_words,
                                                            include_words_regexs=self.include_words_regexs,
                                                            exclude_words_regexs=self.exclude_words_regexs,
                                                            lemmatize=self.lemmatize,
                                                            lemmatization_strategy=self.lemmatization_strategy)

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
                    logging.log(logging.INFO, u'File excluded: ' + path)
            elif os.path.islink(path) and os.path.isdir(path):
                # Manage symbolic links to directories
                os.path.walk(path, self._visit_dir_add_to_vector, arg)

    def _process_dir_add_to_vector(self, file_list, dirname, filenames):
        """
        This method adds to a list all the files contained in a given directory

        :param file_list: list of file paths to be rcompleted. This list is modified by the method
        :param dirname: current directory name
        :param filenames: list of file names in current directory

        :return: The modified file_list
        """
        for name in filenames:
            path = os.path.join(dirname, name)
            if os.path.isfile(path):
                if self._matches(name):
                    file_list.append(path)
                else:
                    logger.info('File excluded: ' + path)
        return file_list

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
            # Check whether file should be excluded
            for r in self.exc_reg_exps:
                if (r.match(s) != None):
                    return False
        return True

    def _get_paths(self, directory):
        """
        Get the paths of the all the files contained in a given directory.

        :param directory: The folder to explore.
        """
        paths = []

        for _directory, _dirnames, _filenames in os.walk(directory):
            self._process_dir_add_to_vector(paths, _directory, _filenames)
        return paths

    def _read_file_text(self, path):
        if self.encoding is None:
            with open(path, "r") as f:
                return f.read()
        else:
            with codecs.open(path, "r", self.encoding) as f:
                return f.read()
            return f.read()

    def _process_text_file(self, path):
        logger.debug('Processing file {}/{}: '.format(self.counter, len(self.paths), path))
        start_time = time.time()

        t1 = time.time()
        if self.encoding is None:
            f = open(path, "r")
        else:
            f = codecs.open(path, "r", self.encoding)
        text = '\n'.join(f.readlines())
        f.close()
        t2 = time.time()
        logger.debug('Read file in {} ms'.format((t2 - t1) * 1000))

        t1 = time.time()
        res = self.preprocessor.preprocess_text(text,
                                                remove_stop_words=self.remove_stop_words,
                                                include_words_regexs=self.include_words_regexs,
                                                exclude_words_regexs=self.exclude_words_regexs,
                                                lemmatize=self.lemmatize,
                                                lemmatization_strategy=self.lemmatization_strategy)

        t2 = time.time()
        logger.debug('Preprocessed text in {} ms'.format((t2 - t1) * 1000))

        t1 = time.time()
        res = self.preprocessor.remove_empty_strings(res)
        t2 = time.time()
        logger.debug('Remove empty strings in {} ms'.format((t2 - t1) * 1000))

        end_time = time.time()
        logger.debug('Preprocessed file {}/{}: in {} milliseconds'.format(self.counter,
            len(self.paths), path, (end_time - start_time) * 1000))

        return res


class SentenceDocIterator(object):
    """
    This class iterates along the text contained in one files. Text processing includes:
      0. Sentence segmentation
      1. Normalize
      2. Tokenize
      3. Remove stop words
    As a result, a vector containing normalizaed words is returned on every iteration.
    """

    def __init__(self,
                 path: Text,
                 text_preprocessor: TextPreprocessor,
                 sentence_segmenter: SentenceSegmenter,
                 remove_stop_words: bool = True,
                 inc: Iterable[Text] = None,
                 exc: Iterable[Text] = None,
                 include_words_regexs: Iterable[Text] = None,
                 exclude_words_regexs: Iterable[Text] = None,
                 lemmatize: bool = False,
                 lemmatization_strategy: Text = LemmatizationStrategy.ALL,
                 encoding: Text = None):
        """
        :param path: Path of the file to process.
        :param text_preprocessor: Instance of altran.text_preprocessing.TextPreprocessor.TextPreprocessor class
        :param sentence_segmenter:
        :param remove_stop_words: Whether to remove stop words
        :param inc: List of regular expressions matching the files to be included into
            the processing. If left None, all files will be included.
        :param exc: List of regular expressions matching the files to be excluded into
            the processing among those considered as included in the inc parameter. If left
            None, no file will be excluded.
        :param include_words_regexs: List of regular expressions (strings) that must be matched
            by any document word.
        :param exclude_words_regexs: List of regular expressions (strings) that must not be
            matched by any document word.
        :param lemmatize: Whether lemmatize the text. Default, False.
        :param lemmatization_strategy: Strategy to use in lemmatization (see altran.nlp.lemma.const.py)
        :param encoding:
        """
        # Create temp dir
        self.temp_dir = tempfile.mkdtemp()
        shutil.copyfile(path, os.path.join(self.temp_dir, os.path.split(path)[-1]))
        self.iterator = SentenceDocTreeIterator(self.temp_dir,
                                                text_preprocessor,
                                                sentence_segmenter,
                                                remove_stop_words=remove_stop_words,
                                                inc=inc,
                                                exc=exc,
                                                include_words_regexs=include_words_regexs,
                                                exclude_words_regexs=exclude_words_regexs,
                                                lemmatize=lemmatize,
                                                lemmatization_strategy=lemmatization_strategy,
                                                encoding=encoding)

    def __iter__(self):
        for s in self.iterator:
            yield s
