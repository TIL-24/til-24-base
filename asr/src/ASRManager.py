import torch
import torchaudio
import io

# FACEBOOK VERSION
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
class ASRManager:
    def __init__(self):
        self.model_name = "facebook/wav2vec2-large-960h"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)

    def transcribe(self, audio_bytes: bytes) -> str:
        audio_tensor, sampling_rate = torchaudio.load(io.BytesIO(audio_bytes))
        
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            audio_tensor = resampler(audio_tensor)
            sampling_rate = 16000
        
        # Process the audio file
        inputs = self.processor(audio_tensor.squeeze(0), sampling_rate=sampling_rate, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        # Perform the transcription
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        
        return transcription[0]  # Return the first element of the batch
    
"""
# OPENAI VERSION
from transformers import WhisperForConditionalGeneration, WhisperProcessor
class ASRManager:
    def __init__(self):
        self.model_name = "openai/whisper-large-v3"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

    def transcribe(self, audio_bytes: bytes) -> str:
        audio_tensor, sampling_rate = torchaudio.load(io.BytesIO(audio_bytes))
        
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            audio_tensor = resampler(audio_tensor)
            sampling_rate = 16000
        
        # Process the audio file
        input_features = self.processor(audio_tensor.squeeze(0), sampling_rate=sampling_rate, return_tensors="pt").input_features
        input_features = input_features.to(self.device)
        
        # Perform the transcription
        with torch.no_grad():
            generated_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        print(transcription)

        return transcription[0]  # Return the first element of the batch
"""