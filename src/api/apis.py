import json.scanner
import requests
import json
import time

from dotenv import dotenv_values
# Load environment variables from .env file
env_vars = dotenv_values(".env")
HF_TOKEN = env_vars.get("HF_TOKEN")

class HuggingFaceAPI:
    def __init__(self) -> None:
        self.HF_TOKEN = HF_TOKEN
        self.headers = {"Authorization": f"Bearer {self.HF_TOKEN}"}
        
        self.TRANSCRIPTION_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
        self.SUMMARIZATION_URL = "https://api-inference.huggingface.co/models/Falconsai/text_summarization"
        self.FALCON_INSTRUCT_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"


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
    
    def get_instruct_summary(self, text):
        prompt = f"""
        Summarize the following dialogue delimited by triple backquotes:
        ```{text}```

        Summary:
        """
        payload = {
            "inputs": prompt
            }
        response = requests.post(self.FALCON_INSTRUCT_URL, headers=self.headers, json=payload).json()
        result = response[0]['generated_text'].split('Summary:\n')[-1]
        return result
    

if __name__ == "__main__":
    huggingface_api = HuggingFaceAPI()
    # text = huggingface_api.get_transcription("audio.mp3")['text']
    # print("Transcribed text: ", str(text))
    # summary = huggingface_api.get_summarization(text)
    # print("Summarized text: ", summary[0]['summary_text'])

    dialogue1 = """
        SPEAKER_01:  Are we, we're not allowed to dim the lights so everyone can see that a bit better.
        SPEAKER_00:  Yeah.
        SPEAKER_01:  Okay.
        SPEAKER_01:  That's fine.
        SPEAKER_01:  Am I supposed to be standing up there?
        SPEAKER_00:  So we've got both of these clipped on.
        SPEAKER_01:  Is she going to answer me? Yeah, I've got it.
        SPEAKER_01:  Both of them.
        SPEAKER_00:  Good. This is going to fall off. Okay. Hello everybody. I'm Sarah, the project manager, and this is our first meeting, surprisingly enough.
        SPEAKER_01:  Okay, this is our agenda. We will do some stuff, get to know each other a bit better, feel more comfortable with each other. Then we'll go do tool training, talk about the
        SPEAKER_01:  project plan, discuss our own ideas and everything. And we've got 25 minutes to do that as far
        SPEAKER_01:  as I can understand. Now we're developing a remote control, which
        """
    
    result = huggingface_api.get_instruct_summary(dialogue1)
    # print(result)
    print(result[0]['generated_text'].split("Summary:\n")[-1])