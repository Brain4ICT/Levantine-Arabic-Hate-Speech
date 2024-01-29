import pickle
import streamlit as st
import re
st.title("Best model for Levantine Arabic Hate Speech Detection - TALLIP Journal")
text=st.text_input("Leave your Levantine comment here ...",value="")
cleaned_text = re.sub('[^؀-ۿ]+', ' ', str(text))
cleaned_text = re.sub('\s+', ' ', cleaned_text).strip()
vectorizer=pickle.load(open("best-l-vect.pickle","rb"))
x=vectorizer.transform([cleaned_text]).toarray()
model=pickle.load(open("best-l-model.pickle","rb"))
pred=model.predict(x)
if st.button("Check Hate Speech"):
                       st.success(pred)
    
