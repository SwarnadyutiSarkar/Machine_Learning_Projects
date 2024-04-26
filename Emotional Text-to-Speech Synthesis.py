import pyttsx3

def text_to_speech(text, emotion):
    # Initialize pyttsx3 engine
    engine = pyttsx3.init()

    # Set properties for the speech
    if emotion == 'happy':
        engine.setProperty('rate', 150)  # Adjust speaking rate for happiness
        engine.setProperty('volume', 1.0)  # Adjust volume for happiness
    elif emotion == 'sad':
        engine.setProperty('rate', 100)  # Adjust speaking rate for sadness
        engine.setProperty('volume', 0.5)  # Adjust volume for sadness
    elif emotion == 'angry':
        engine.setProperty('rate', 200)  # Adjust speaking rate for anger
        engine.setProperty('volume', 1.0)  # Adjust volume for anger

    # Convert text to speech
    engine.say(text)
    engine.runAndWait()

# Example usage
text = "I am feeling happy today"
emotion = 'happy'
text_to_speech(text, emotion)
