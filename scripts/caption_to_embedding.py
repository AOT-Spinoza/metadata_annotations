### WE WILL USE ANOTHER ENVIRONMENT FOR STATISTICAL ANALYSIS OF THE DATA 
### THIS SCRIPT USES THE ENVIRONMENT "metadata_stats"


import gensim.downloader as api
import numpy as np
import torch
import string

def transform_captions(caption, model):
    caption = caption.translate(str.maketrans('', '', string.punctuation)).lower()
    words = caption.split()
    caption_embedding = np.zeros(model.vector_size)
    count = 0
    for word in words:
        if word in model.key_to_index:
            caption_embedding += model[word]
            count += 1
        else:
            print(f'Word not in vocabulary: {word}')  # Add this line
            
    if count > 0:
        caption_embedding /= count
    return caption_embedding  # Return the array directly