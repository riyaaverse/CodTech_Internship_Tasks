import pyttsx3

# Initialize the Text-to-Speech engine
engine = pyttsx3.init()

# The text you want to convert to audio
text = "Hello, this is a test for my internship task brother. I am checking if the code works properly."

print("Generating audio file...")

# Save the audio to a file named 'sample_audio.wav'
engine.save_to_file(text, 'sample_audio.wav')

# Run the engine to process the command
engine.runAndWait()

print("Done! 'sample_audio.wav' has been created in your folder.")