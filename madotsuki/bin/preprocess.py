#!/usr/bin/env python3

from __future__ import division, print_function, absolute_import

from docopt import docopt
import numpy as np

from nnmnkwii.datasets import FileDataSource, FileSourceDataset
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.preprocessing.f0 import interp1d
from nnmnkwii.util import apply_delta_windows
from nnmnkwii.io import hts

from os.path import join as joinpath
from glob import glob

import pysptk

from scipy.io import wavfile
from tqdm import tqdm

import os
from os.path import basename, splitext, exists

import sys


order = 59
frame_period = 5
windows = [
        (0, 0, np.array([1.0]))
        (1, 1, np.array([-0.5, 0.0, 0.5]))
        (1, 1, np.array([1.0, -2.0, 1.0]))
]


class LinguisticSource(FileDataSource):
    def __init__(self,
                 data_root
                 add_frame_features=False,
                 subphone_features=None,
                 question_path=None):
        self.data_root = data_root
        self.add_frame_features = add_frame_features
        self.subphone_features = subphone_features
        self.test_paths = None

        if question_path is None:
            raise RuntimeError("Please specify a question path!")
        self.binary_dict, self.continuous_dict = hts.load_question_set(question_path)

    def collect_files(self):
        files = sorted(glob(joinpath(self.data_root,
                                     "**", "*.lab")))

        return files

    def collect_features(self, path):
        labels = hts.load(path)
        features = fe.linguistic_features(labels,
                                          self.binary_dict,
                                          self.continuous_dict,
                                          add_frame_features=self.add_frame_features,
                                          subphone_features=self.subphone_features)

        if self.add_frame_features:
            indices = labels.silence_frame_indices().astype(np.int)
        else:
            indices = labels.silence_frame_indices()

        features = np.delete(features, indices, axis=0)

        return features.astype(np.float32)


class DurationFeatureSource(FileDataSource):
    def __init__(self, data_root):
        self.data_root = data_root

    def collect_files(self):
        files = sorted(glob(joinpath(self.data_root,
                                     "**", "*.lab")))
        return files

    def collect_features(self, path):
        labels = hts.load(path)
        features = fe.duration_features(labels)
        indices = labels.silence_phone_indices()
        features = np.delete(features, indices, axis=0)

        return features.astype(np.float32)


class AcousticSource(FileDataSource):
    def __init__(self, data_root):
        self.data_root = data_root

    def collect_files(self):
        wav_paths = sorted(glob(joinpath(self.data_root,
                                         "**", "*.wav")))
        labels = sorted(glob(joinpath(self.data_root,
                                      "**", "*.lab")))

        return wav_paths, label_paths

    def collect_features(self, wav_path, label_path):
        fs, x = wavfile.read(wav_path)
        x = x.astype(np.float64)

        f0, timeaxis = pyworld.dio(x, fs, frame_period=frame_period)
        f0 = pyworld.stonemask(x, f0, timeaxis, fs)
        spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
        aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)

        bap = pyworld.code_aperiodicity(aperiodicity, fs)
        mgc = pysptk.sp2mc(spectrogram, order=order,
                           alpha=pysptk.util.mcepalpha(fs))

        f0 = f0[:, None]
        lf0 = f0.copy()
        nonzero_indices = np.nonzero(f0)
        lf0[nonzero_indices] = np.log(f0[nonzero_indices])

        vuv = (lf0 != 0).astype(np.float32)
        lf0 = interp1d(lf0, kind="slinear")

        mgc = apply_delta_windows(mgc, windows)
        lf0 = apply_delta_windows(lf0, windows)
        bap = apply_delta_windows(bap, windows)

        features = np.hstack((mgc, lf0, vuv, bap))

        labels = hts.load(label_path)
        features = features[:labels.num_frames()]
        indices = labels.silence_frame_indices()
        features = np.delete(features, indices, axis=0)

        return features.astype(np.float32)


@hydra.main(version_base=None,
            config_path="conf",
            config_name="data")
def my_app(cfg):
    data_root = cfg.data_root
    question_path = cfg.question_path
    overwrite = True # TODO

    DST_ROOT = "data"

    # X -> Y
    # X: linguistic
    # Y: duration
    X_duration_source = LinguisticSource(data_root,
                                         add_frame_features=False,
                                         subphone_features=None,
                                         question_path=question_path)
    Y_duration_source = DurationFeatureSource(data_root)

    X_duration = FileSourceDataset(X_duration_source)
    Y_duration = FileSourceDataset(Y_duration_source)

    # X -> Y
    # X: linguistic
    # Y: acoustic
    subphone_features = "coarse_coding"
    X_acoustic_source = LinguisticSource(data_root,
                                         add_frame_features=True,
                                         subphone_features=subphone_features,
                                         question_path=question_path)
    Y_acoustic_source = AcousticSource(data_root)

    X_acoustic = FileSourceDataset(X_acoustic_source)
    Y_acoustic = FileSourceDataset(Y_acoustic_source)

    X_duration_root = joinpath(DST_ROOT, "X_duration")
    Y_duration_root = joinpath(DST_ROOT, "Y_duration")
    X_acoustic_root = joinpath(DST_ROOT, "X_acoustic")
    Y_acoustic_root = joinpath(DST_ROOT, "Y_acoustic")

    skip_duration_feature_extraction = exists(X_duration_root) and exists(Y_duration_root)
    skip_acoustic_feature_extraction = exists(X_acoustic_root) and exists(Y_acoustic_root)

    if overwrite:
        skip_acoustic_feature_extraction = False
        skip_duration_feature_extraction = False

    for d in [X_duration_root, Y_duration_root, X_acoustic_root, Y_acoustic_root]:
        if not os.path.exists(d):
            print("mkdirs: {}".format(d))
            os.makedirs(d)

    if not skip_duration_feature_extraction:
        print("Duration linguistic feature dim", X_duration[0].shape)
        print("Duration feature dim", Y_duration[0].shape)
        for idx, (x, y) in tqdm(enumerate(zip(X_duration, Y_duration))):
            name = splitext(basename(X_duration.collected_files[idx][0]))[0]

            xpath = joinpath(X_duration_root, name + ".bin")
            ypath = joinpath(Y_duration_root, name + ".bin")

            x.tofile(xpath)
            y.tofile(ypath)
    else:
        print("Features for duration model training found, skipping feature extraction...")

    if not skip_acoustic_feature_extraction:
        print("Acoustic linguistic feature dim", X_acoustic[0].shape)
        print("Acoustic feature dim", Y_acoustic[0].shape)
        for idx, (x, y) in tqdm(enumerate(zip(X_acoustic, Y_acoustic))):
            name = splitext(basename(X_acoustic.collected_files[idx][0]))[0]

            xpath = joinpath(X_acoustic_root, name + ".bin")
            ypath = joinpath(Y_acoustic_root, name + ".bin")

            x.tofile(xpath)
            y.tofile(ypath)
    else:
        print("Features for acoustic model training found, skipping feature extraction...")

    sys.exit(0)


if __name__ == "__main__":
    my_app()
