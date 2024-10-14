import os
# setup cache path for huggingface
os.environ['CACHE_DIR'] = '/mnt/users_scratch/astitva/CACHE/'
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['HF_HOME'] = os.environ['CACHE_DIR']
os.environ['HF_DATASETS_CACHE'] = os.environ['CACHE_DIR']
os.environ['TRANSFORMERS_CACHE']= os.environ['CACHE_DIR']


from transformers import AutoProcessor, AutoModelForCTC, Wav2Vec2Processor
import librosa
import torch
from itertools import groupby
from datasets import load_dataset

def decode_phonemes(
    ids: torch.Tensor, processor: Wav2Vec2Processor, ignore_stress: bool = False
) -> str:
    """CTC-like decoding. First removes consecutive duplicates, then removes special tokens."""
    # removes consecutive duplicates
    ids = [id_ for id_, _ in groupby(ids)]

    special_token_ids = processor.tokenizer.all_special_ids + [
        processor.tokenizer.word_delimiter_token_id
    ]
    # converts id to token, skipping special tokens
    phonemes = [processor.decode(id_) for id_ in ids if id_ not in special_token_ids]

    # joins phonemes
    prediction = " ".join(phonemes)

    # whether to ignore IPA stress marks
    if ignore_stress == True:
        prediction = prediction.replace("ˈ", "").replace("ˌ", "")

    return prediction

checkpoint = "bookbot/wav2vec2-ljspeech-gruut"

model = AutoModelForCTC.from_pretrained(checkpoint)
processor = AutoProcessor.from_pretrained(checkpoint)
sr = processor.feature_extractor.sampling_rate

# # load dummy dataset and read soundfiles
# ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
# audio_array = ds[0]["audio"]["array"]


# or, read a single audio file
wav_path = 'audios/stress_test.wav'
audio_array, _ = librosa.load(wav_path, sr=sr)

inputs = processor(audio_array, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs["input_values"]).logits

predicted_ids = torch.argmax(logits, dim=-1)
prediction = decode_phonemes(predicted_ids[0], processor, ignore_stress=True)
breakpoint()


from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
model_name = "facebook/wav2vec2-base-960h"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

outputs = model(**inputs, labels=predicted_ids)

attention_weights = outputs.logits[0].squeeze(0).cpu().detach().numpy()
# Get the aligned phonemes
aligned_phonemes = []
for i in range(len(attention_weights)):
    # Get the index of the maximum attention weight
    max_index = np.argmax(attention_weights[i])
    
    # Check if the maximum attention weight is above a certain threshold
    if attention_weights[i][max_index] > 0.5:
        aligned_phonemes.append((i, phonemes[max_index]))
    else:
        aligned_phonemes.append((i, None))
# Print the aligned phonemes
for frame_number, phoneme in aligned_phonemes:
    if phoneme is not None:
        print(f"Frame {frame_number}: {phoneme}")
    else:
        print(f"Frame {frame_number}: No phoneme aligned")
