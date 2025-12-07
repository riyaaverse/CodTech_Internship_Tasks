# Task 2: Speech Recognition System
# Instructions: Build a basic speech-to-text system using pre-trained models.
# Libraries used: SpeechRecognition, PyAudio

# NOTE: You must install the libraries first. In your terminal run:
# pip install SpeechRecognition pyaudio

import speech_recognition as sr

def transcribe_audio(file_path):
    """
    Takes a .wav audio file path and returns the transcribed text.
    """
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    try:
        # Load the audio file
        # The file must be in the same folder as this script
        with sr.AudioFile(file_path) as source:
            print(f"Loading '{file_path}'...")
            
            # Listen to the data (load audio to memory)
            audio_data = recognizer.record(source)
            
            print("Transcribing... (sending to Google Speech API)")
            
            # Recognize speech using Google's free API
            text = recognizer.recognize_google(audio_data)
            return text
            
    except sr.UnknownValueError:
        return "Error: Google Speech Recognition could not understand the audio."
    except sr.RequestError as e:
        return f"Error: Could not request results from Google Speech Recognition service; {e}"
    except FileNotFoundError:
        return f"Error: The file '{file_path}' was not found. Please make sure the file name is correct."

if __name__ == "__main__":
    # IMPORTANT: You need a .wav file to test this.
    # 1. Record a short audio clip on your phone or computer.
    # 2. Convert it to .wav format (there are many free online converters).
    # 3. Name it 'sample_audio.wav' and put it in this same folder.
    
    audio_filename = "sample_audio.wav" 
    
    print("\n--- Speech to Text System ---")
    result = transcribe_audio(audio_filename)
    
    print("\n--- Transcription Result ---")
    print(result)