from src.api.diarize import SpeakerDiarizationPipeline

if __name__ == "__main__":
    diarizer = SpeakerDiarizationPipeline.from_pretrained()

    audio_path = "test1.wav"
    result = diarizer(audio_path)
    print(result)