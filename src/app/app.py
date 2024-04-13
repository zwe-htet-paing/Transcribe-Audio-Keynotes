import gradio as gr


class AudioTranscriber:
    def __init__(self):
        self.title = "Transribe Audio KeyNotes ⚡️"
        self.description = """
        The system can transcribe audio to text and extract keynotes from the audio. The system uses the Whisper large-v2 model by OpenAI and speaker diarization model by pyannote.
        """
        self.article = "Whisper large-v2 model by OpenAI. Speaker diarization model by pyannote. 🚀"

    def transcribe(self, audio, task, group_by_speaker):
        if audio is None:
            return "No audio detected. Please try again."
        if task == "transcribe":
            return "Transcription not available for this task. Please select 'transcribe' or 'keynote'."

    def create_interface(self, source, label):
        return gr.Interface(
            fn=self.transcribe,
            inputs=[
                gr.Audio(sources=source, label=label, type="filepath"),
                gr.Radio(["transcribe", "keynote"], label="Task", value="transcribe"),
                gr.Checkbox(value=True, label="Group by speaker"),
            ],
            outputs=[
                gr.Textbox(label="Transcription", show_copy_button=True),
                # download the transcript
                gr.File(label="Download Transcript"),
            ],
            allow_flagging="never",
            title=self.title,
            description=self.description,
            article=self.article,
        )

    def launch(self):
        microphone = self.create_interface("microphone", None)
        audio_file = self.create_interface("upload", "Audio file")

        demo = gr.Blocks()
        with demo:
            gr.TabbedInterface([microphone, audio_file], ["Microphone", "Audio file"])
        demo.queue(max_size=10)
        demo.launch()


if __name__ == "__main__":
    transcriber = AudioTranscriber()
    transcriber.launch()