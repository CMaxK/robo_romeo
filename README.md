# Robo-Romeo

![image](https://user-images.githubusercontent.com/103648207/175067813-01692494-a157-495d-878a-24c2fe23c356.png)

- Applied Object-Oriented Programming (OOP) to design the batch for training dataset.
- Utilised CNN model (EfficientNetB0) to encode images into vectors and added embedded layer to tokenize the captions corresponding with images.
- Established the LSTM model and trained it using Google Cloud Platform (GCP) Vertex AI to predict the next word of sequences and output whole sentences. 
- Built up the scoring function which using doc2vec to transfer sentences into vectors and calculate the cosine similarities to evaluate the performance of image captioning. 
- Imported open API called GPT-3 to output the beautiful poetry according to information gathered from images. 
- Developed a website using Streamlit to present both poetry and robot voices. 



# Introduction - can AI be creative?

Some intro words on problem, solution, techniques

# Solution structure

Neural networks used:
- CNN (EfficientNetB0) : Image encoding
- LSTM : Word embedding and sequencing
- GTP3 : Output poetry

Training on over 118k images and cpations

Tasks split in 2 parts:
- look through image and generate caption
- turn caption into poetry


# Bonus - Attention layer

Some words on this special attention layer that we added

<img width="716" alt="Screenshot 2022-06-21 at 17 34 46" src="https://user-images.githubusercontent.com/103648207/174852206-2bf930da-ae4c-4293-bb1a-7818eaa1ab00.png">
<img width="615" alt="Screenshot 2022-06-21 at 17 35 26" src="https://user-images.githubusercontent.com/103648207/174852319-342c0405-ee32-453c-bb2d-09981d645493.png">

# Datasets used

<img width="177" alt="Screenshot 2022-06-22 at 16 44 04" src="https://user-images.githubusercontent.com/103648207/175074550-c72df250-b26a-4974-81af-759467e95958.png">

ImageNet - image database designed for use in computer vision research

# Output predictions

Examples of captions generated

# Performance metrics

- Doc2vec for transfering sentences to vectors

- Cosine similarities as scores
 
<img width="376" alt="Screenshot 2022-06-22 at 16 13 51" src="https://user-images.githubusercontent.com/103648207/175067109-e4a1c8e4-5a75-4bc5-835b-785c377e1e57.png">

# Final product

- Link to Streamlit
- Link to Streamlit GitHub
- Link to demo presentation slides



# Website

Link : [https://awesome-github-readme-profile.netlify.app](https://share.streamlit.io/cmaxk/robo_romeo_streamlit/app.py)


# Our Robo-Romeo's Output


<img width="1715" alt="Screenshot 2022-06-21 at 17 24 45" src="https://user-images.githubusercontent.com/103648207/174849984-cfd70617-4a2f-498d-b9e8-ec978ce8d439.png">

<img width="1717" alt="Screenshot 2022-06-21 at 17 44 47" src="https://user-images.githubusercontent.com/103648207/174853791-b04d34f6-3e49-41ca-ace8-51138948287b.png">

