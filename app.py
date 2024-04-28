import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

from tqdm import tqdm  # Import tqdm for progress bar

import time # for time taken calc

# Load pre-trained model
label_dict = {"Urgency": 0, "Not Dark Pattern": 1, "Scarcity": 2, "Misdirection": 3, "Social Proof": 4, "Obstruction": 5, "Sneaking": 6, "Forced Action": 7}
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_dict))

# Load fine-tuned weights
fine_tuned_model_path = "F:/backup-kali/codeFiles/projects/cognigaurd/fine_tuned_bert2/finetuned_BERT_epoch_5.model"
model.load_state_dict(torch.load(fine_tuned_model_path, map_location=torch.device('cpu')))

# Preprocess the new text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Function to map numeric label to dark pattern name
def get_dark_pattern_name(label):
    reverse_label_dict = {v: k for k, v in label_dict.items()}
    return reverse_label_dict[label]

def find_dark_pattern(text_predict):
    encoded_text = tokenizer.encode_plus(
        text_predict,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    # Making the predictions
    model.eval()

    with torch.no_grad():
        inputs = {
            'input_ids': encoded_text['input_ids'],
            'attention_mask': encoded_text['attention_mask']
        }
        outputs = model(**inputs)

    predictions = outputs.logits

    # Post-process the predictions
    probabilities = torch.nn.functional.softmax(predictions, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()

    return get_dark_pattern_name(predicted_label)

# Streamlit app
def main():

    # navigation
    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link("pages/page_1.py", label="Training Metrics", icon="1️⃣")
    # st.page_link("pages/page_2.py", label="Page 2", icon="2️⃣")
    st.page_link("https://github.com/4darsh-Dev/CogniGaurd", label="GitHub", icon="🌎")
    # Set page title
    st.title("Dark Pattern Detector")

    # Display welcome message
    st.write("Welcome to Dark Pattern Detector powered by CogniGuard")
    
    #
    st.write("#### Built with Fine-Tuned BERT and Hugging Face Transformers")
    

    # Get user input
    text_to_predict = st.text_input("Enter the text to find Dark Pattern")

    if st.button("Predict"):
        # Record the start time
        start_time = time.time()

        # Add a simple progress message
        st.write("Predicting Dark Pattern...")


        progress_bar = st.progress(0)

        for i in tqdm(range(10), desc="Predicting", unit="prediction"):
            predicted_darkp = find_dark_pattern(text_to_predict)
            progress_bar.progress((i + 1) * 10)
            time.sleep(0.5)  # Simulate some processing time

        # Record the end time
        end_time = time.time()

        # Calculate the total time taken
        total_time = end_time - start_time

        # Display the predicted dark pattern and total time taken
        st.write(f"Result: {predicted_darkp}")
        st.write(f"Total Time Taken: {total_time:.2f} seconds")

        


        # Add footer
        st.markdown('<p style="text-align:center;">Made with ❤️ by <a href="https://www.adarshmaurya.onionreads.com">Adarsh Maurya</a></p>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()

