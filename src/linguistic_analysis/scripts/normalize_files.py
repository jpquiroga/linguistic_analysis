import argparse
import logging

from linguistic_analysis.text_iterators import SentenceSegmenter, SentenceDocTreeIterator, SentenceSegmenterNLTK
from linguistic_analysis.text_preprocessing import TextPreprocessor


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", help="Input path to files")
    parser.add_argument("out_path", help="Path to the output file")
    parser.add_argument("nltk_folder", help="Path to the NLTK data folder")
    parser.add_argument("languages", help="Comma separated languages. Example: french,spanish,english")
    return parser

if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    parser = get_argparser()
    args = parser.parse_args()

    in_path = args.in_path
    out_path = args.out_path
    nltk_folder = args.nltk_folder
    languages = args.languages.split(",")

    print("Starting normalization with parameters: {}".format(args))
#    sentence_segmenter = SentenceSegmenter({"en": None, "fr": None, "es": None})
    sentence_segmenter = SentenceSegmenterNLTK()

    text_preprocessor = TextPreprocessor(nltk_folder, languages)
    sentence_doc_iterator = SentenceDocTreeIterator(in_path,
                                                    text_preprocessor,
                                                    sentence_segmenter,
                                                    remove_stop_words=False)

    print("Normalizing sentences...")
    counter = 0
    with open(out_path, "w") as f:
        for s in sentence_doc_iterator:
            f.write(" ".join(s))
            f.write("\n")
            counter += 1
            if counter % 1000 == 0:
                print("Normalized sentences: {}".format(counter))
