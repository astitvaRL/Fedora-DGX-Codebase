from elevenlabs import play, save
from elevenlabs.client import ElevenLabs, VoiceSettings
import os

save_name = 'feedback'

#text="Hello Hello Hello Hello Mic Testing Mic Testing Mic Testing one one one one two two two two three three three three four four four four five five five five six six six six seven seven seven seven eight eight eight eight nine nine nine ten ten ten ten Finish finish finish."
#text="Hello, this is a demo for automatic facial animation for hand-drawn characters. Do you think that it matches the drawing style? How about the lip-sync quality? Please share your feedback. Thankyou!"
#text="The sun was shining brightly in the clear blue sky, and a gentle breeze rustled through the leaves of the tall trees. The sound of birds chirping and water flowing created a soothing melody that filled the air. As I walked along the winding path, I noticed the vibrant colors of the flowers and the intricate patterns on the rocks. The scent of freshly cut grass wafted through the air, reminding me of summertime and carefree days."
text="Do butterflies know they're wearing the prettiest dresses in the garden? Why do cats purr? Are they telling us a secret? Do raindrops race each other to the ground to see who wins?"
voice="Alice"

client = ElevenLabs(
  api_key="sk_434d8785e25c01e426cb87fc7c807e5826bb6828280eeb3e", # Defaults to ELEVEN_API_KEY
)

voice_settings = VoiceSettings(stability=1.0, similarity_boost=0.75)

audio = client.generate(
  text=text,
  voice=voice,
  model="eleven_multilingual_v2",
  output_format="mp3_22050_32",
  voice_settings=voice_settings
)

play(audio)


audio = client.generate(
  text=text,
  voice=voice,
  model="eleven_multilingual_v2",
  output_format="mp3_22050_32",
  voice_settings=voice_settings
)

#save audio to file as .wav
os.makedirs(f'./inputs/{save_name}', exist_ok=True)
save_filepath = f'./inputs/{save_name}/{save_name}'
save(audio, f"{save_filepath}.wav")
f = open(f"{save_filepath}.txt", "w")
f.write(text)
f.close()
breakpoint()
