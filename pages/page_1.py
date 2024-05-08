import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    st.title("Training Metrics")
    

    st.markdown("#### Sweeps for hyperparameter tuning")
    st.image("assets/img/cogni-bert-12sweeps.png", use_column_width=True )

    data = [
    {"val_loss": 0.60, "train_loss": 1.24, "val_f1": 0.82},
    {"val_loss": 0.37, "train_loss": 0.44, "val_f1": 0.89},
    {"val_loss": 0.39, "train_loss": 0.23, "val_f1": 0.88},
    {"val_loss": 0.351, "train_loss": 0.13, "val_f1": 0.90},
    {"val_loss": 0.353, "train_loss": 0.071, "val_f1": 0.922},
    

    ]

    # Convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(data) 

    df["epoch"] = range(1, len(data)+1 )
    
    st.markdown("### Cogni-BERT Best Model Train")


    # st.dataframe(data)  


    col1, col2, col3 = st.columns(3)

    # Line chart for validation loss with Seaborn
    with col1:
        st.markdown("#### Validation Loss")
        fig, ax = plt.subplots()
        sns.lineplot(x="epoch", y="val_loss", data=df, ax=ax, color='skyblue', marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.xlabel("epoch")
        plt.ylabel("Validation Loss")
        st.pyplot(fig)

    # Line chart for training loss with Seaborn
    with col2:
        st.markdown("#### Training Loss")
        fig, ax = plt.subplots()
        sns.lineplot(x="epoch", y="train_loss", data=df, ax=ax, color='salmon', marker='s', linestyle='--', linewidth=2, markersize=8)
        plt.xlabel("epoch")
        plt.ylabel("Training Loss")
        st.pyplot(fig)

    # Line chart for F1 score with Seaborn
    with col3:
        st.markdown("#### Validation F1")
        fig, ax = plt.subplots()
        sns.lineplot(x="epoch", y="val_f1", data=df, ax=ax, color='limegreen', marker='D', linestyle='-.', linewidth=2, markersize=8)
        plt.xlabel("epoch")
        plt.ylabel("F1 Score")
        st.pyplot(fig)


    st.markdown('<p style="text-align:center;">Made with ❤️ by <a href="https://www.adarshmaurya.onionreads.com">Adarsh Maurya</a></p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
