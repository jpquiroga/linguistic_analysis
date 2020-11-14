from gensim.models import Word2Vec
#from gensim.models.word2vec import PathLineSentences
import logging
import os.path as path
from typing import Iterable

DEFAULT_SIZE = 100
DEFAULT_WINDOW = 5
DEFAULT_MIN_COUNT = 2
DEFAULT_ALPHA = 0.025
DEFAULT_SEED = 1
DEFAULT_MAX_VOCAB_SIZE = None
DEFAULT_SAMPLE = 1e-3
DEFAULT_WORKERS = 3
DEFAULT_NEGATIVE = 5
DEFAULT_ITER = 5
DEFAULT_BATCH_WORDS = 10000
DEFAULT_MIN_ALPHA = 0.0001


logger = logging.getLogger(__name__)

"""
    sentences = None, corpus_file = None, size = 100, alpha = 0.025, window = 5, min_count = 5,
    max_vocab_size = None, sample = 1e-3, seed = 1, workers = 3, min_alpha = 0.0001,
    sg = 0, hs = 0, negative = 5, ns_exponent = 0.75, cbow_mean = 1, hashfxn = hash, iter = 5, null_word = 0,
    trim_rule = None, sorted_vocab = 1, batch_words = MAX_WORDS_IN_BATCH, compute_loss = False, callbacks = (),
    max_final_vocab = None
"""

def generate_word2vec_model_old(model_name, dir_path, text_iterator,
                            min_count=DEFAULT_MIN_COUNT,
                            size=DEFAULT_SIZE, window=DEFAULT_WINDOW, alpha=DEFAULT_ALPHA, seed=DEFAULT_SEED,
                            max_vocab_size=DEFAULT_MAX_VOCAB_SIZE, sample=DEFAULT_SAMPLE, workers=DEFAULT_WORKERS,
                            min_alpha=DEFAULT_MIN_ALPHA, sg=0, hs=0, negative=DEFAULT_NEGATIVE, cbow_mean=1, hashfxn=hash,
                            iter=DEFAULT_ITER, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=DEFAULT_BATCH_WORDS):
    """
    Generate a Word2Vec model and save it to disk.

    Args:
        - model_name: Name of the model to be generated. The generated model will be saved in file with this name.
        - dir_path: Path to the directory where the generated model will be saved.
        - text_iterator: Object used to iterate over the lines of text to be analyzed.
        - text_preprocessor:
        - is_tweet: True if the text to be analyzed comes from a Twitter message; False otherwise.
        - min_count: Minimum number of words to be considered into the word2vec models (words appearing less than
            min_count will not be taken into account); i.e., ignore all words with total frequency lower than this.
        - size: The dimensionality of the feature vectors.
        - alpha: The initial learning rate (will linearly drop to zero as training progresses).
        - seed: For the random number generator. Initial vectors for each
            word are seeded with a hash of the concatenation of word + str(seed).
        - max_vocab_size: limit RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types
            need about 1GB of RAM. Set to `None` for no limit (default).
        - sample: Threshold for configuring which higher-frequency words are randomly downsampled;
            default is 1e-3, useful range is (0, 1e-5).
        - workers: Use this many worker threads to train the model (=faster training with multicore machines).
        - min_alpha: Minimum alpha value.
        - sg: Defines the training algorithm. By default (`sg=0`), CBOW is used.
            Otherwise (`sg=1`), skip-gram is employed.
        - hs: = if 1, hierarchical softmax will be used for model training.
            If set to 0 (default), and `negative` is non-zero, negative sampling will be used.
        - negative: = if > 0, negative sampling will be used, the int for negative
            specifies how many "noise words" should be drawn (usually between 5-20).
            Default is 5. If set to 0, no negative samping is used.
        - cbow_mean: = if 0, use the sum of the context word vectors. If 1 (default), use the mean.
            Only applies when cbow is used.
        - hashfxn: hash function to use to randomly initialize weights, for increased
            training reproducibility. Default is Python's rudimentary built in hash function.
        - iter: number of iterations (epochs) over the corpus.
        - trim_rule: vocabulary trimming rule, specifies whether certain words should remain
            in the vocabulary, be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used), or a callable that accepts parameters (word, count, min_count) and
            returns either util.RULE_DISCARD, util.RULE_KEEP or util.RULE_DEFAULT.
            Note: The rule, if given, is only used prune vocabulary during build_vocab() and is not stored as part
            of the model.
        - sorted_vocab: = if 1 (default), sort the vocabulary by descending frequency before
            assigning word indexes.
        - batch_words: = target size (in words) for batches of examples passed to worker threads (and
            thus cython routines). Default is 10000. (Larger batches can be passed if individual
            texts are longer, but the cython code may truncate.)

    Returns:
        A gensim Word2Vec model built on the data.

    """

    logger.info("Generating Word2Vec model " + model_name)
    model = Word2Vec(text_iterator, min_count=min_count, size=size, alpha=alpha, window=window,
                     max_vocab_size=max_vocab_size, sample=sample, seed=seed, workers=workers, min_alpha=min_alpha,
                     sg=sg, hs=hs, negative=negative, cbow_mean=cbow_mean, hashfxn=hashfxn, iter=iter, null_word=null_word,
                     trim_rule=trim_rule, sorted_vocab=sorted_vocab, batch_words=batch_words)
    if model:
        model_file = path.join(dir_path, model_name)
        model.save(model_file)
        logger.info("Word2Vec model " + model_name + " has been successfully generated at path " + dir_path)
    return model


def generate_word2vec_model(sentences: Iterable, vector_size=100, window=5, min_count=1, workers=4):
    model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=1, workers=4)
    model.train()
    # TODO






#__WORD2VEC_GEN_SECTION = "WORD2VEC_GEN"

