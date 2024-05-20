from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
import io

class ASRManager:
    def __init__(self):
        self.model_name = "facebook/wav2vec2-base-960h"
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to("cuda")

    def transcribe(self, audio_bytes: bytes) -> str:
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        inputs = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        return transcription[0]

