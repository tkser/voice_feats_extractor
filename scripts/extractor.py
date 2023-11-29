import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from voice_feats_extractor.configs.preprocess import Config
from voice_feats_extractor.label import Label
from voice_feats_extractor.modules.tacotron2.layers import TacotronSTFT
from voice_feats_extractor.wave import Wave

logging.basicConfig(level=logging.INFO)


def extractor(cfg: Config) -> None:
    """Extract features from lab files and wav files

    Args:
        cfg (Config): Config object
    """
    logging.info("Start extracting features")

    output_dir = Path(cfg.preprocess.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    current_dir = Path.cwd()
    logging.info(f"Current directory: {current_dir}")

    wav_files = sorted(current_dir.glob(cfg.preprocess.wav_glob))
    lab_files = sorted(current_dir.glob(cfg.preprocess.lab_glob))

    logging.info(f"Number of wav files: {len(wav_files)}")
    logging.info(f"Number of lab files: {len(lab_files)}")

    if cfg.preprocess.strict_lab and len(wav_files) == len(lab_files):
        msg = "Number of wav and lab files are the same."
        raise ValueError(msg)

    output_dirs = {
        "duration": output_dir / "duration",
        "pitch": output_dir / "pitch",
        "mel": output_dir / "mel",
        "accent": output_dir / "accent",
    }

    for output_dir in output_dirs.values():
        output_dir.mkdir(exist_ok=True)

    n_frames = 0

    stft = TacotronSTFT(
        filter_length=cfg.stft.filter_length,
        hop_length=cfg.stft.hop_length,
        win_length=cfg.stft.win_length,
        n_mel_channels=cfg.mel.n_mel_channels,
        sampling_rate=cfg.wave.sampling_rate,
        mel_fmin=cfg.mel.mel_fmin,
        mel_fmax=cfg.mel.mel_fmax,
    )

    for i, lab_file in enumerate(tqdm(lab_files, total=len(wav_files), desc="Preprocess")):
        lab_id = lab_file.stem

        wav_file = wav_files[0].parent / f"{lab_id}.wav"
        wav_id = lab_id

        if cfg.preprocess.strict_lab:
            wav_file = wav_files[i]
            wav_id = wav_file.stem

        if wav_id != lab_id:
            msg = f"wav_id ({wav_id}) and lab_id ({lab_id}) are different."
            raise ValueError(msg)

        label = Label.load_from_path(lab_file, sec_unit=cfg.label.sec_unit)
        _, durations, start_time, end_time = label.get_alignments(
            sampling_rate=cfg.wave.sampling_rate,
            hop_length=cfg.stft.hop_length,
        )

        wave = Wave.load_from_path(
            wav_file,
            sampling_rate=cfg.wave.sampling_rate,
            max_wave_value=cfg.wave.max_wav_value,
        )
        wave = wave.clip(start_time=start_time, end_time=end_time)

        pitch = wave.get_pitch(hop_length=cfg.stft.hop_length, durations=durations)

        mel = wave.get_mel(stft=stft, durations=durations)

        accents = list(label.get_accent_ids())

        n_frames += mel.shape[1]

        duration_file = output_dirs["duration"] / f"{wav_id}.npy"
        pitch_file = output_dirs["pitch"] / f"{wav_id}.npy"
        mel_file = output_dirs["mel"] / f"{wav_id}.npy"
        accent_file = output_dirs["accent"] / f"{wav_id}.npy"

        np.save(duration_file, durations)
        np.save(pitch_file, pitch)
        np.save(mel_file, mel.T)
        np.save(accent_file, accents)

    logging.info(f"Number of frames: {n_frames}")
    logging.info(f"Total time: {n_frames * cfg.stft.hop_length / cfg.wave.sampling_rate / 3600:.2f} hours")
    logging.info("Finish extracting features")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default="conf/default_config.yaml")
    args = parser.parse_args()

    base_cfg = OmegaConf.structured(Config)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, cfg)

    extractor(cfg)
