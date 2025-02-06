import requests
import streamlit as st
from environs import Env

if 'response' not in st.session_state:
    st.session_state.response = None

if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

env = Env()
env.read_env(override=True)

@st.cache_data
def submit_request(text_input):
    response = requests.post(env.str("ENDPOINT"), json={
        "raw_text": text_input,
        "scenario": "bert-delexicalized"
    })

    try:
        response.raise_for_status()        
    except requests.exceptions.HTTPError as e:
        st.error(f"Error: {e}")
        st.session_state.response = None
        return
    
    st.session_state.input_text = ""
    st.session_state.response = response.json()


st.title("Portuguese Text Delexicalizer")
st.markdown("---")

st.session_state.input_text = st.text_area("The Portuguese Text You want to Test", placeholder="Type here ...")

# if text_input is less than 50 characters. Raise an warning

if len(st.session_state.input_text) < 20:
    st.warning("The text is too short. Please type more than 20 characters")
  
if len(st.session_state.input_text) > 300:
    st.error("The text is too long. Please type less than 300 characters")

if len(st.session_state.input_text) > 20 and len(st.session_state.input_text) < 300:
    st.button("Test the Text", on_click=submit_request, args=(st.session_state.input_text,))

# Add horizontal line
st.markdown("---")

if st.session_state.response:    
    col1, col2 = st.columns([0.4,0.6], vertical_alignment="center", gap="medium")
    
    with col1:
        container = st.container(border=True)
        container.html(f"<h4>Original Text</h4>")
        container.write(st.session_state.response['raw_text'])
        container.write("---")
        #container.write(st.session_state.response['delexicalized_text'])
        container.html(f"<h4>Delexicalized Text</h4>")
        container.write(st.session_state.response['delexicalized_text'])


    with col2:
        st.bar_chart(
            {
                "European Portuguese": st.session_state.response['european_portuguese'],
                "Brazilian Portuguese": st.session_state.response['brazilian_portuguese']
            }, 
            horizontal=True, 
            height=300,
            color="#FF4D00",
            use_container_width=True
        )

#TODO: Introduce Visualization of dataset stats