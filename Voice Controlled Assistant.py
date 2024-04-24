import speech_recognition as sr

# Initialize the recognizer
r = sr.Recognizer()

# Function to convert speech to text
def listen_to_speech():
    with sr.Microphone() as source:
        print("Say something:")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text
        except:
            print("Sorry, I could not understand that.")
            return ""

# Main loop
while True:
    command = listen_to_speech().lower()
    if "hello" in command:
        print("Hello! How can I assist you?")
    elif "exit" in command:
        print("Goodbye!")
        break
    else:
        print("I didn't catch that. Could you please repeat?")
