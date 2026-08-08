"""Convert model3 (build_model3) weights from an .h5 file to the native .keras format.

Usage:
    python convert_model3_to_keras.py [weights.h5] [output.keras]

Defaults to ./models/exp3multif0.h5 if no weights path is given, and derives
the output path from the input filename (extension replaced with .keras) if
no output path is given.
"""

import argparse
import os

from models import build_model3

DEFAULT_WEIGHTS_PATH = os.path.join("models", "exp3multif0.h5")


def convert(h5_path, keras_path):
    model = build_model3()
    model.load_weights(h5_path)
    model.save(keras_path)
    print("Saved {}".format(keras_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "weights_path",
        nargs="?",
        default=DEFAULT_WEIGHTS_PATH,
        help="Path to model3 .h5 weights file (default: {})".format(DEFAULT_WEIGHTS_PATH),
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default=None,
        help="Path to output .keras file (default: same name with .keras extension)",
    )
    args = parser.parse_args()

    output_path = args.output_path
    if output_path is None:
        root, _ = os.path.splitext(args.weights_path)
        output_path = root + ".keras"

    convert(args.weights_path, output_path)


if __name__ == "__main__":
    main()
