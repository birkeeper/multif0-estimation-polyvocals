"""
Predict multiple F0 output from input audio or folder.
This code is linked to the ISMIR paper:

Helena Cuesta, Brian McFee and Emilia Gómez (2020).
Multiple F0 Estimation in Vocal Ensembles using Convolutional Neural Networks.
In Proceedings of the 21st International Society for Music Information Retrieval Conference (ISMIR).
Montreal, Canada (virtual).
"""

from __future__ import print_function
import models
import utils
import utils_train

import numpy as np
import tensorflow as tf

import os
import argparse

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

CHUNK_LEN = 2000


def save_salience_map(salience, save_path, model_name, thresh, weights_path=None):
    """Store the raw (freq, time) salience map so it can be re-analysed without
    re-running inference -- e.g. sweeping the threshold offline, or reading the
    salience value at a known F0 instead of a binary detected/not-detected.

    Saved as compressed float16 (salience is in [0, 1], so ~3 decimal digits is
    ample) together with the frequency and time grids needed to interpret it.
    """
    freq_grid = utils.get_freq_grid()
    time_grid = utils.get_time_grid(salience.shape[1])
    np.savez_compressed(
        save_path,
        salience=salience.astype(np.float16),
        freq_grid=freq_grid, time_grid=time_grid,
        model_name=model_name, thresh=thresh,
        weights_path=weights_path if weights_path is not None else '',
    )


def get_single_test_prediction_phase_free(model, audio_file=None):
    """Generate output from a model given an input numpy file
    """

    if audio_file is not None:
        # should not be the case
        pump = utils.create_pump_object()
        features = utils.compute_pump_features_segmented(pump, audio_file)
        input_hcqt = features['dphase/mag'][0]


    else:
        raise ValueError("one of npy_file or audio_file must be specified")

    input_hcqt = input_hcqt.transpose(1, 2, 0)[np.newaxis, :, :, :]

    n_t = input_hcqt.shape[3]
    t_slices = list(np.arange(0, n_t, CHUNK_LEN))
    output_list = []
    # we need two inputs
    for t in t_slices:
        p = model.predict(np.transpose(input_hcqt[:, :, :, t:t+CHUNK_LEN], (0, 1, 3, 2)))[0, :, :]

        output_list.append(p)

    predicted_output = np.hstack(output_list)
    return predicted_output, input_hcqt

def get_single_test_prediction(model, audio_file=None):
    """Generate output from a model given an input numpy file.
       Part of this function is part of deepsalience
    """

    if audio_file is not None:

        pump = utils.create_pump_object()
        features = utils.compute_pump_features_segmented(pump, audio_file)
        input_hcqt = features['dphase/mag'][0]
        input_dphase = features['dphase/dphase'][0]

    else:
        raise ValueError("One audio_file must be specified")

    input_hcqt = input_hcqt.transpose(1, 2, 0)[np.newaxis, :, :, :]
    input_dphase = input_dphase.transpose(1, 2, 0)[np.newaxis, :, :, :]

    n_t = input_hcqt.shape[3]
    t_slices = list(np.arange(0, n_t, CHUNK_LEN))
    output_list = []

    for t in t_slices:
        p = model.predict([np.transpose(input_hcqt[:, :, :, t:t+CHUNK_LEN], (0, 1, 3, 2)),
                           np.transpose(input_dphase[:, :, :, t:t+CHUNK_LEN], (0, 1, 3, 2))]
                          )[0, :, :]

        output_list.append(p)

    predicted_output = np.hstack(output_list)
    return predicted_output, input_hcqt, input_dphase


def main(args):

    model_name = args.model_name
    audiofile = args.audiofile
    audio_folder = args.audio_folder
    plot_salience = args.plot_salience
    save_salience = args.save_salience

    # load model weights
    if model_name == 'model1':

        save_key = 'exp1multif0'
        model_path = "./models/{}.h5".format(save_key)
        model = models.build_model1()
        model.load_weights(model_path)
        thresh = 0.4

    elif model_name == 'model2':

        save_key = 'exp2multif0'
        model_path = "./models/{}.h5".format(save_key)
        model = models.build_model2()
        model.load_weights(model_path)
        thresh = 0.5

    elif model_name == 'model3':

        save_key = 'exp3multif0'
        model_path = "./models/{}.h5".format(save_key)
        model = models.build_model3()
        model.load_weights(model_path)
        thresh = 0.5

    elif model_name == 'model4':

        save_key = 'exp4multif0'
        model_path = "./models/{}.h5".format(save_key)
        model = models.build_model3()
        model.load_weights(model_path)
        thresh = 0.4

    elif model_name == 'model7':

        save_key = 'exp7multif0'
        model_path = "./models/{}.h5".format(save_key)
        model = models.build_model3_mag()
        model.load_weights(model_path)
        thresh = 0.4

    else:
        raise ValueError(
            "Specified model must be one of: model1, model2, model3, "
            "model4, model7.")

    # allow overriding the model's default global threshold from the CLI
    if args.thresh is not None:
        thresh = args.thresh

    # allow overriding the model's default weights file, so different
    # checkpoints of the same architecture can be compared side by side
    if args.model_weights is not None:
        model_path = args.model_weights
        model.load_weights(model_path)

    # label identifying both the architecture and the weights used, so
    # outputs from different checkpoints don't overwrite each other and
    # stay distinguishable when comparing plots/CSVs side by side
    weights_label = os.path.splitext(os.path.basename(model_path))[0]
    label = '{}_{}'.format(model_name, weights_label)

    # compile model

    model.compile(
        loss=utils_train.bkld, metrics=['mse', utils_train.soft_binary_accuracy],
        optimizer='adam'
    )
    print("Model compiled")

    # select operation mode and compute prediction
    if audiofile != "0":

        if model_name == 'model7':
            # predict using trained model
            predicted_output, _ = get_single_test_prediction_phase_free(
                model, audio_file=os.path.join(
                    audio_folder, audiofile)
            )
        else:
            # predict using trained model
            predicted_output, _, _ = get_single_test_prediction(
                model, audio_file=audiofile
            )

        predicted_output = predicted_output.astype(np.float32)

        est_times, est_freqs = utils_train.pitch_activations_to_mf0(predicted_output, thresh)

        stem = os.path.splitext(audiofile)[0]

        if plot_salience:
            utils_train.plot_salience(
                predicted_output, '{}_{}_salience.png'.format(stem, label),
                est_times, est_freqs, model_name=label
            )

        if save_salience:
            path = '{}_{}_salience.npz'.format(stem, label)
            save_salience_map(predicted_output, path, model_name, thresh, weights_path=model_path)
            print(" > > > Salience map saved as {}.".format(path))

        # rearrange output
        for i, (tms, fqs) in enumerate(zip(est_times, est_freqs)):
            if any(fqs <= 0):
                est_freqs[i] = np.array([f for f in fqs if f > 0])

        output_path = '{}_{}.csv'.format(stem, label)
        utils_train.save_multif0_output(est_times, est_freqs, output_path)

        print(" > > > Multiple F0 prediction for {} exported as {}.".format(
            audiofile, output_path)
        )

    elif audio_folder != "0":

        for audiofile in os.listdir(audio_folder):

            if not audiofile.endswith('wav'): continue

            if model_name == 'model7':
                # predict using trained model
                predicted_output, _ = get_single_test_prediction_phase_free(
                    model, audio_file=os.path.join(
                        audio_folder, audiofile)
                )

            else:

                # predict using trained model
                predicted_output, _, _ = get_single_test_prediction(
                    model, audio_file=os.path.join(
                        audio_folder, audiofile)
                )

            predicted_output = predicted_output.astype(np.float32)

            est_times, est_freqs = utils_train.pitch_activations_to_mf0(predicted_output, thresh)

            stem = os.path.splitext(audiofile)[0]

            if plot_salience:
                utils_train.plot_salience(
                    predicted_output,
                    save_path=os.path.join(
                        audio_folder, '{}_{}_salience.png'.format(stem, label)
                    ),
                    est_times=est_times, est_freqs=est_freqs,
                    model_name=label
                )

            if save_salience:
                path = os.path.join(
                    audio_folder, '{}_{}_salience.npz'.format(stem, label)
                )
                save_salience_map(predicted_output, path, model_name, thresh, weights_path=model_path)
                print(" > > > Salience map saved as {}.".format(path))

            # rearrange output
            for i, (tms, fqs) in enumerate(zip(est_times, est_freqs)):
                if any(fqs <= 0):
                    est_freqs[i] = np.array([f for f in fqs if f > 0])

            output_path = os.path.join(
                audio_folder, '{}_{}.csv'.format(stem, label)
            )
            utils_train.save_multif0_output(est_times, est_freqs, output_path)

            print(" > > > Multiple F0 prediction for {} exported as {}.".format(
                audiofile, output_path)
            )
    else:
        raise ValueError("One of audiofile and audio_folder must be specified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict multiple F0 output of an input audio file or all the audio files inside a folder.")

    parser.add_argument("--model",
                        dest='model_name',
                        type=str,
                        help="Specify the ID of the model "
                             "to use for the prediction: model1 (Early/Deep) / "
                             "model2 (Early/Shallow) / "
                             "model3 (Late/Deep, recommended)")

    parser.add_argument("--model_weights",
                        dest='model_weights',
                        default=None,
                        type=str,
                        help="Path to a weights file (.h5 / .weights.h5) overriding the "
                             "model's default checkpoint, so different fine-tuned "
                             "weights of the same architecture can be compared. The "
                             "weights filename is included in the salience plot title, "
                             "the salience map, and the output CSV filename.")

    parser.add_argument("--audiofile",
                        dest='audiofile',
                        default="0",
                        type=str,
                        help="Path to the audio file to analyze. If using the folder mode, this should be skipped.")

    parser.add_argument("--audio_folder",
                        dest='audio_folder',
                        default="0",
                        type=str,
                        help="Directory with audio files to analyze. If using the audiofile mode, this should be skipped.")

    parser.add_argument("--plot_salience",
                        dest='plot_salience',
                        action='store_true',
                        help="If set, save a pitch salience (time-frequency) plot as a PNG "
                             "next to each output CSV, before it is converted into F0 estimates.")

    parser.add_argument("--save_salience",
                        dest='save_salience',
                        action='store_true',
                        help="If set, save the raw (frequency x time) salience map as a "
                             "compressed .npz next to each output CSV, alongside its "
                             "frequency and time grids. Lets the prediction be "
                             "re-analysed later -- e.g. sweeping the threshold or "
                             "reading the salience at a known F0 -- without re-running "
                             "the model.")

    parser.add_argument("--thresh",
                        dest='thresh',
                        default=None,
                        type=float,
                        help="Override the global salience threshold for peak selection. "
                             "If omitted, the model's default is used. Ignored when --adaptive_thresh is set.")

    main(parser.parse_args())
