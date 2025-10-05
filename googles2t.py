import speech_recognition as sr

def speech_to_text_from_mic():
    """
    Captures audio from the microphone and converts it to text.
    """
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Adjusting for ambient noise, please wait...")
        # Listen for a second to adjust the energy threshold for ambient noise
        recognizer.adjust_for_ambient_noise(source)
        
        print("\nSay something!")
        
        try:
            # Listen for the user's input
            audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
            print("Recognizing...")
            
            # Use Google's Speech Recognition
            # The recognize_google() method uses the Google Web Speech API
            text = recognizer.recognize_google(audio_data)
            print(f"You said: {text}")

        except sr.WaitTimeoutError:
            print("Listening timed out. No speech was detected.")
        except sr.UnknownValueError:
            # This error is raised when Google's API could not understand the audio
            print("Google Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            # This error is raised for issues with the API request (e.g., no internet)
            print(f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    speech_to_text_from_mic()
