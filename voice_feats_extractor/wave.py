from pathlib import Path

import numpy as np
import pyworld as pw
import torch
from librosa.util import normalize
from scipy.interpolate import interp1d
from scipy.io.wavfile import read as read_wav

from voice_feats_extractor.modules.tacotron2.layers import TacotronSTFT


class Wave:
    def __init__(self, wave: np.ndarray, sampling_rate: int) -> None:
        self.wave = wave
        self.sampling_rate = sampling_rate

    def __len__(self) -> int:
        return len(self.wave)

    def __str__(self) -> str:
        return f"<Wave idx={self.__hash__()} len={len(self)}>"

    @staticmethod
    def load_from_path(
        wav_path: Path,
        sampling_rate: int | None = None,
        max_wave_value: float = 32768.0,
        normalize_rate: float = 0.95,
    ) -> "Wave":
        sr, wave = read_wav(wav_path)

        wave = wave / max_wave_value
        wave = normalize(wave) * normalize_rate
        wave = wave.astype(np.float32)

        if sampling_rate is not None and sr != sampling_rate:
            msg = f"Sampling rate of {wav_path} is {sr}, but {sampling_rate} is given."
            raise ValueError(msg)

        return Wave(wave, sr)

    def clip(self, start_time: float, end_time: float) -> "Wave":
        start_index = int(np.round(start_time * self.sampling_rate))
        end_index = int(np.round(end_time * self.sampling_rate))
        self.wave = self.wave[start_index:end_index]
        return self

    def get_pitch(self, hop_length: int, durations: list[int]) -> np.ndarray:
        pitch, t = pw.dio(
            self.wave.astype(np.float64),
            self.sampling_rate,
            frame_period=hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(self.wave.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(durations)]

        if np.sum(pitch != 0) <= 1:
            msg = "Number of non-zero pitch values is less than or equal to 1."
            raise ValueError(msg)

        nonzeros = np.where(pitch != 0)[0]
        interp_fn = interp1d(
            nonzeros,
            pitch[nonzeros],
            fill_value=(pitch[nonzeros[0]], pitch[nonzeros[-1]]),
            bounds_error=False,
        )
        pitch = interp_fn(np.arange(0, len(pitch)))

        _pos = 0
        for m, d in enumerate(durations):
            if d > 0:
                pitch[m] = np.mean(pitch[_pos : _pos + d])
            else:
                pitch[m] = 0
            _pos += d
        pitch = pitch[: len(durations)]

        return pitch

    def get_mel(self, stft: TacotronSTFT, durations: list[int]) -> np.ndarray:
        mel = self._get_mel_from_stft(stft)
        mel = mel[:, : sum(durations)]
        return mel

    def _get_mel_from_stft(self, stft: TacotronSTFT) -> np.ndarray:
        wav = torch.from_numpy(self.wave).unsqueeze(0)
        wav = torch.autograd.Variable(wav, requires_grad=False)
        mel = stft.mel_spectrogram(wav)
        mel = mel.squeeze(0)
        mel = mel.numpy().astype(np.float32)
        return mel
