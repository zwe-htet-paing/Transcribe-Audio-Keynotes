from src.api.diarize import ASRDiarizationPipeline

if __name__ == "__main__":
    diarizer = ASRDiarizationPipeline.from_pretrained()

    audio_path = "test1.wav"
    result = diarizer(audio_path)
    print(result)