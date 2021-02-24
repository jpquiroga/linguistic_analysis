from gensim.models import Word2Vec
import argparse
import glob
import logging
import os
from tqdm import tqdm
from typing import Text
import yaml
import sys


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Yaml file containing training information")
    return parser

def count_words(path: Text) -> int:
    counter = 0
    with open(path) as f:
        for line in f.readlines():
            words = line.replace("\t", " ").strip().split(" ")
            counter += len(words)
    return counter

if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    parser = get_argparser()
    args = parser.parse_args()

    config_file = args.config_file
    with open(config_file) as f:
        config = yaml.load(f, yaml.FullLoader)

    # Train one model for every pre-processed document
    for f_path in tqdm(glob.glob(os.path.join(config["text_folder"], "*.txt"))):
        # Train a model
        print("\n\nTraining for {}".format(f_path))
        model = Word2Vec(corpus_file=f_path, sg=config["sg"], window=config["window"],
                         min_count=config["min_count"], workers=config["workers"])
        model.train(corpus_file=f_path, epochs=config["num_epochs"], total_examples=model.corpus_count,
                    total_words=count_words(f_path))
        # Save the model
        # Create the destination folder
        f_name = os.path.split(f_path)[-1][:-4]
        base_dest_folder = os.path.join(config["model_destination_dir"], f_name)
        model_name = "model"
        model_dir_name = "{}_w2v_{}_w{}_mincount{}_ep".format(model_name,
                                                              "cbow" if config["sg"] == 1 else "sg",
                                                              config["window"],
                                                              config["min_count"],
                                                              config["num_epochs"])
        model_destination_dir = os.path.join(base_dest_folder, model_dir_name)
        os.makedirs(model_destination_dir, exist_ok=False)
#        os.makedirs(os.path.join(model_destination_dir, model_dir_name), exist_ok=False)

        model_destination = os.path.join(model_destination_dir, model_dir_name + ".model")
        model.wv.save(model_destination)

        # Write to TSV
        with open(os.path.join(model_destination_dir, model_dir_name + ".tsv"), "w") as f:
            for w, v in model.wv.vocab.items():
                f.write("\t".join([str(v) for v in model.wv[w]]))
                f.write("\n")
        with open(os.path.join(model_destination_dir, model_dir_name + "_metadata" + ".tsv"), "w") as f:
            f.write("label\tcount\n")
            for w, v in model.wv.vocab.items():
                f.write("\t".join([w, str(v.count)]))
                f.write("\n")
