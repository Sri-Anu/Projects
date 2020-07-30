from gtts import gTTS #voice
import os #audio file 
def text2speech():
    text = input("What do you want the audio to say?")
    tts = gTTS(text=text, lang='en') 
    save_file = input(r"Enter the path on where you want to save it. Get rid of the quotation marks please!")
    name = input("What do you want to name the file? Don't put .mp3 just the name.")
    tts.save(save_file+name+".mp3") 
    print("File saved in"+save_file+name+".mp3")
text2speech()