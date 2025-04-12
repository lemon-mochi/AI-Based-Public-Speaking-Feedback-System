# AI-Based-Public-Speaking-Feedback-System
Final project for CMPT 419 (Affective Computing)

## Required packages
numpy, matplotlib, pandas, librosa, audioread, pydub, soundfile, seaborn, sklearn, ffmpeg

numpy must be version 2.1 because numba is not supported in later versions.

## Link to Data.zip
The Data.zip file is too large for Github so it can be downloaded from here.
https://1sfu-my.sharepoint.com/:u:/g/personal/aap9_sfu_ca/EcBeCT5WlANGk1kmI4hOZhEBrUhwYIWPfARiv5l2jx3zQg?e=KBePTN

## Running the app ##
Enter in the terminal,
```
streamlit run app.py
```

## Self-Evaluation
Overall, our project achieved its primary goal of providing meaningful feedback on speech delivery through an AI-driven system. While some planned components were deferred or modified due to resource constraints. 

We shifted from unsupervised clustering to supervised learning after observing limited variation in early datasets, ultimately combining CREMA-D, TED Talks, and SEP-28k datasets and introducing manual annotations to improve label quality. A Random Forest classifier was selected for its interpretability and strong performance. A significant portion of our time was spent in finding a good dataset to improve model accuracy. This effort should be considered when assessing our progress relative to the original plan. Despite challenges with clustering and dataset limitations, we adapted effectively by diversifying datasets, refining features, and improving labeling strategies.

We view this project as a successful step toward creating an accessible tool for improving public speaking skills and are confident that it can be expanded further in future iterations.
