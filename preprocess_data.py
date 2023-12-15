import os
import shutil
import glob
import argparse
from tqdm import tqdm
import json

import sentencepiece as spm
import numpy as np

from bhw2.src.utils.utils import PAD_ID, UNK_ID, BOS_ID, EOS_ID


def main(args):
    assert args.vocab_size > 0, "Vocab size must be positive"
    if os.path.exists(f"{args.vocab_size}.model"):
        os.remove(f"{args.vocab_size}.model")
    if os.path.exists(f"{args.vocab_size}.vocab"):
        os.remove(f"{args.vocab_size}.vocab")
    if os.path.exists(f"{args.output_dir}"):
        shutil.rmtree(f"{args.output_dir}/")

    print(f"limit: {args.limit}")
    os.makedirs(args.output_dir, exist_ok=True)
    input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))[: args.limit]

    for file_name in tqdm(input_files):
        output_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(file_name))[0]}.txt")
        with open(file_name) as fin:
            data = json.load(fin)

        with open(output_path, "w") as fout:
            for text in tqdm(data, "json -> txt"):
                fout.write(text["story"] + "\n")
    print(f"Files are in {args.output_dir}.")

    print("Training a vocab...")
    spm.SentencePieceTrainer.train(
        input=",".join(glob.glob(os.path.join(args.output_dir, "*.txt"))),
        model_prefix=str(args.vocab_size),
        vocab_size=args.vocab_size,
        model_type="bpe",
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
    )
    tokenizer = spm.SentencePieceProcessor(model_file=f"{args.vocab_size}.model")

    print("Preparing the dataset...")
    os.makedirs(args.output_dir, exist_ok=True)
    tokenized = []
    idx = 0
    idxs = []
    for file in tqdm(input_files, "json -> npy"):
        with open(file) as fin:
            data = json.load(fin)
        for text in data:
            tokenized.append(tokenizer.encode(text["story"]))
            idxs.append(np.array([idx, idx + len(tokenized[-1])]))
            idx += len(tokenized[-1]) + 1

    val_part = 10000
    np.save(os.path.join(args.output_dir, "train.npy"), np.concatenate(tokenized[:-val_part]).astype(np.int16))
    np.save(os.path.join(args.output_dir, "train_idxs.npy"), np.stack(idxs[:-val_part]).astype(np.int64))

    np.save(os.path.join(args.output_dir, "val.npy"), np.concatenate(tokenized[-val_part:]).astype(np.int16))
    np.save(os.path.join(args.output_dir, "val_idxs.npy"), np.stack(idxs[-val_part:]).astype(np.int64))

    print(f"Dataset is saved in {args.output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=str, default="TinyStories_all_data", help="input directory path")
    parser.add_argument("-o", "--output-dir", type=str, default="data", help="output directory path")
    parser.add_argument("-v", "--vocab-size", type=int, default=4096, help="vocabulary size")
    parser.add_argument("-l", "--limit", type=int, default=100, help="limit input_files from the TinyStories_all_data")
    args = parser.parse_args()
    main(args)
