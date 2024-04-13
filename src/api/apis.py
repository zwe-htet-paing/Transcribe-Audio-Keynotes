import json.scanner
import requests
import json
import time

class HuggingFaceAPI:
    def __init__(self) -> None:
        self.HF_TOKEN = "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"
        self.headers = {"Authorization": f"Bearer {self.HF_TOKEN}"}
        
        self.TRANSCRIPTION_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
        self.SUMMARIZATION_URL = "https://api-inference.huggingface.co/models/Falconsai/text_summarization"


    def get_transcription(self, filename):
        with open(filename, "rb") as f:
            data = f.read()
        while True:
            try:
                time.sleep(1)
                response = requests.post(self.TRANSCRIPTION_URL, headers=self.headers, data=data)
                break
            except Exception:
                continue

        return response.json()
    
    def get_summarization(self, text):
        payload = {
            'inputs': text
        }
        while True:
            try:
                time.sleep(1)
                response = requests.post(self.SUMMARIZATION_URL, headers=self.headers, json=json.dumps(payload))
                break
            except Exception:
                continue

        return response.json()
    

if __name__ == "__main__":
    huggingface_api = HuggingFaceAPI()
    text = huggingface_api.get_transcription("audio.mp3")['text']
    print("Transcribed text: ", str(text))
    summary = huggingface_api.get_summarization(text)
    print("Summarized text: ", summary[0]['summary_text'])