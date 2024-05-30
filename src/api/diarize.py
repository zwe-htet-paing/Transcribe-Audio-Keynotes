"""
# Huggingface models
- https://huggingface.co/openai/whisper-medium
- https://huggingface.co/pyannote/speaker-diarization-3.0
"""

from typing import List, Optional, Union

import numpy as np
import requests
import torch
from pyannote.audio import Pipeline
from torchaudio import functional as F
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read

import warnings

warnings.filterwarnings("ignore")

from dotenv import dotenv_values

# Load environment variables from .env file
env_vars = dotenv_values(".env")
HF_TOKEN = env_vars.get("HF_TOKEN")

from huggingface_hub import login

login(token=HF_TOKEN)


class ASRDiarizationPipeline:
    def __init__(self, asr_pipeline, diarization_pipeline):
        self.asr_pipeline = asr_pipeline
        self.sampling_rate = self.asr_pipeline.feature_extractor.sampling_rate

        self.diarization_pipeline = diarization_pipeline

    @classmethod
    def from_pretrained(
        cls,
        asr_model="openai/whisper-medium",
        diarization_model="pyannote/speaker-diarization-3.0",
        chunk_length_s=30,
        use_auth_token=True,
        device="cuda",
    ):
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=asr_model,
            chunk_length_s=chunk_length_s,
            device=f"{device}:0",
            token=use_auth_token,
        )
        diarization_pipeline = Pipeline.from_pretrained(
            diarization_model, use_auth_token=use_auth_token
        )
        diarization_pipeline.to(torch.device(device))

        return cls(asr_pipeline, diarization_pipeline)

    def postprocess_diarization(self, diarization_result):
        segments = []
        for segment, track, label in diarization_result.itertracks(yield_label=True):
            segments.append(
                {
                    "segment": {"start": segment.start, "end": segment.end},
                    "track": track,
                    "label": label,
                }
            )

        new_segments = []
        prev_segment = cur_segment = segments[0]
        for i in range(1, len(segments)):
            cur_segment = segments[i]

            # check if there changed speaker ("label")
            if cur_segment["label"] != prev_segment["label"] and i < len(segments):
                # add the start/end times for the super-segment to the new list
                new_segments.append(
                    {
                        "segment": {
                            "start": prev_segment["segment"]["start"],
                            "end": cur_segment["segment"]["start"],
                        },
                        "speaker": prev_segment["label"],
                    }
                )
                prev_segment = segments[i]

        # add the last segment(s) if there was no speaker change
        new_segments.append(
            {
                "segment": {
                    "start": prev_segment["segment"]["start"],
                    "end": cur_segment["segment"]["end"],
                },
                "speaker": prev_segment["label"],
            }
        )

        return new_segments

    def merge_trancription(self, segments, asr_out, group_by_speaker=True):
        transcript = asr_out["chunks"]

        # get the end timestamps for each chunk from the ASR output
        end_timestamps = np.array([chunk["timestamp"][-1] for chunk in transcript])

        segmented_preds = []
        # align the diarizer timestamps and the ASR timestamps
        for segment in segments:
            # get the diarizer end timestamp
            end_time = segment["segment"]["end"]

            # find the ASR end timestamp that is closest to the diarizer's end timestamp and cut the transcript to here
            upto_idx = np.argmin(np.abs(end_timestamps - end_time))

            if group_by_speaker:
                segmented_preds.append(
                    {
                        "speaker": segment["speaker"],
                        "text": "".join(
                            [chunk["text"] for chunk in transcript[: upto_idx + 1]]
                        ),
                        "timestamp": (
                            transcript[0]["timestamp"][0],
                            transcript[upto_idx]["timestamp"][1],
                        ),
                    }
                )
            else:
                for i in range(upto_idx + 1):
                    segmented_preds.append(
                        {"speaker": segment["speaker"], **transcript[i]}
                    )

            # crop the transcripts and timestamp lists according to the latest timestamp (for faster argmin)
            transcript = transcript[upto_idx + 1 :]
            end_timestamps = end_timestamps[upto_idx + 1 :]

            if len(end_timestamps) == 0:
                break

        return segmented_preds

    def __call__(self, inputs, group_by_speaker=True):
        inputs, diarizer_inputs = self.preprocess(inputs)

        diarization = self.diarization_pipeline(
            {"waveform": diarizer_inputs, "sample_rate": self.sampling_rate},
        )
        new_segments = self.postprocess_diarization(diarization)

        asr_out = self.asr_pipeline(
            {"array": inputs, "sampling_rate": self.sampling_rate},
            return_timestamps=True,
        )
        segmented_preds = self.merge_trancription(
            new_segments, asr_out, group_by_speaker=group_by_speaker
        )

        return segmented_preds

    def preprocess(self, inputs):
        if isinstance(inputs, str):
            if inputs.startswith("http://") or inputs.startswith("https://"):
                inputs = requests.get(inputs).content
            else:
                with open(inputs, "rb") as f:
                    inputs = f.read()

        if isinstance(inputs, bytes):
            inputs = ffmpeg_read(inputs, sampling_rate=self.sampling_rate)

        if isinstance(inputs, dict):
            # Accepting `"array"` which is the key defined in `datasets` for better integration
            if not (
                "sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)
            ):
                raise ValueError(
                    "When passing a dictionary to ASRDiarizePipeline, the dict needs to contain a "
                    '"raw" key containing the numpy array representihttps://www.google.com/search?q=ValueError%3A+Soundfile+is+either+not+in+the+correct+format+or+is+malformed.+Ensure+that+the+soundfile+has+a+valid+audio+file+extension+(e.g.+wav%2C+flac+or+mp3)+and+is+not+corrupted.+If+reading+from+a+remote+URL%2C+ensure+that+the+URL+is+the+full+address+to+**download**+the+audio+file.&oq=ValueError%3A+Soundfile+is+either+not+in+the+correct+format+or+is+malformed.+Ensure+that+the+soundfile+has+a+valid+audio+file+extension+(e.g.+wav%2C+flac+or+mp3)+and+is+not+corrupted.+If+reading+from+a+remote+URL%2C+ensure+that+the+URL+is+the+full+address+to+**download**+the+audio+file.&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRg60gEHMTY3ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8#ip=1ng the audio and a "sampling_rate" key, '
                    "containing the sampling_rate associated with that array"
                )

            _inputs = inputs.pop("raw", None)
            if _inputs is None:
                # Remove path which will not be used from `datasets`.
                inputs.pop("path", None)
                _inputs = inputs.pop("array", None)
            in_sampling_rate = inputs.pop("sampling_rate")
            inputs = _inputs
            if in_sampling_rate != self.sampling_rate:
                inputs = F.resample(
                    torch.from_numpy(inputs), in_sampling_rate, self.sampling_rate
                ).numpy()

        if not isinstance(inputs, np.ndarray):
            raise ValueError(
                f"We expect a numpy ndarray as input, got `{type(inputs)}`"
            )
        if len(inputs.shape) != 1:
            raise ValueError(
                "We expect a single channel audio input for ASRDiarizePipeline"
            )

        # diarization model expects float32 torch tensor of shape `(channels, seq_len)`
        diarizer_inputs = torch.from_numpy(inputs).float()
        diarizer_inputs = diarizer_inputs.unsqueeze(0)

        return inputs, diarizer_inputs
