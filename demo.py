"""_summary_

data = {
        "Audio": "/hom/kasdfadf/test.mp4",
        "Language": "english",
        "speaker": ["speaker_1", "speaker_2"],
        "Timestamp": ["2021-09-01 12:00:00", "2021-09-01 12:00:10"],
        "Transcript": ["Hello, how are you doing today?", "I am fine."],
        "task": "Transcribe"
    }
keynote_data = {
        "Audio": "/hom/kasdfadf/test.mp4",
        "Language": "english",
        "KeyNotes": "Hello, how are you doing today?",
        "task": "Note",
    }
"""

from src.app.app import AudioTranscriber

if __name__ == "__main__":
    transcriber = AudioTranscriber()
    transcriber.launch()
