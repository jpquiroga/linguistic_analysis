import argparse
import logging
import os

from linguistic_analysis.text_iterators import SentenceDocIterator, SentenceSegmenterNLTK
from linguistic_analysis.text_preprocessing import TextPreprocessor
from linguistic_analysis.lemma.french_lemmatizers import FrenchLemmatizer

def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", help="Input path to a folder containing the files to process")
    parser.add_argument("out_path", help="Path to the output folder to write the resulting processed files")
    parser.add_argument("nltk_folder", help="Path to the NLTK data folder")
    parser.add_argument("languages", help="Comma separated languages. Example: french,spanish,english")
    parser.add_argument("remove_accents", help="Remove accents", default="yes")
    parser.add_argument("lemmatize", help="Wether to lemmatize the text", default="no")
    return parser

if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    parser = get_argparser()
    args = parser.parse_args()
    print(args)

    in_path = args.in_path
    out_path = args.out_path
    nltk_folder = args.nltk_folder
    languages = args.languages.split(",")
    remove_accents = args.remove_accents.lower() == "yes"
    lemmatize = args.lemmatize.lower() == "yes"

    print("Starting normalization with parameters: {}".format(args))
    sentence_segmenter = SentenceSegmenterNLTK()

    for file_name in os.listdir(in_path):
        text_preprocessor = TextPreprocessor(nltk_folder, languages, rmv_accents=remove_accents)
        sentence_doc_iterator = SentenceDocIterator(os.path.join(in_path, file_name),
                                                    text_preprocessor,
                                                    sentence_segmenter,
                                                    remove_stop_words=False)
        print("Normalizing sentences in {}...".format(file_name))
        lemmatizer = FrenchLemmatizer() if lemmatize else None
        counter = 0
        with open(os.path.join(out_path, file_name), "w") as f:
            for s in sentence_doc_iterator:
                if lemmatize:
                    s = lemmatizer.lemmatize(s)
                f.write(" ".join(s))
                f.write("\n")
                counter += 1
                if counter % 1000 == 0:
                    print("Normalized sentences: {}".format(counter))
