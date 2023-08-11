from dotenv import find_dotenv, load_dotenv
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
import requests
import os
import torch
import soundfile as sf
from datasets import load_dataset
import streamlit as st


load_dotenv(find_dotenv())
TOKEN = os.getenv("HUGGINGFACE_TOKEN")

def img2text(url) :
    img_to_text = pipeline("image-to-text",model="Salesforce/blip-image-captioning-base")
    text = img_to_text(url)
    print(text)
    return text
     
def generate_story(scenario):
    template = """
    You are a story teller;
    Tou can generate a short story based on a simple narrative, the story should be no more than 20 words;

    CONTEXT : {scenario}
    STORY:
    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_llm  = LLMChain(llm=ChatOpenAI(
        model_name="gpt-3.5-turbo", temperature=1), prompt=prompt, verbose=True)
    story = story_llm.predict(scenario=scenario)

    print(story)
    return story

def text2speech(story):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    inputs = processor(text=story, return_tensors="pt")

    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    sf.write("speech.wav", speech.numpy(), samplerate=16000)
    
    


def main():
    st.set_page_config(page_title="img to audio story")

    st.header("Turn img into audio story")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image.", 
                 use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        st.audio("speech.wav")

if __name__ == '__main__':
    main()








#generate_story(img2text("mina.jpg"))
#text2speech(generate_story(img2text("pic.jpg")))