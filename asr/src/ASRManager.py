from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
import io

class ASRManager:
    def __init__(self):
        self.model_name = "facebook/wav2vec2-large-960h"
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)

    def transcribe(self, audio_bytes: bytes) -> str:
        audio_tensor, sampling_rate = torchaudio.load(io.BytesIO(audio_bytes))
        
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            audio_tensor = resampler(audio_tensor)
            sampling_rate = 16000
        
        # Process the audio file
        inputs = self.processor(audio_tensor.squeeze(0), sampling_rate=sampling_rate, return_tensors="pt")
        
        # Perform the transcription
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)

        return transcription[0]  # Return the first element of the batch
