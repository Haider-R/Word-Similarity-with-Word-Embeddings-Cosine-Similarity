#####################################################
# This is a modified version of the function found at
# https://fasttext.cc/docs/en/english-vectors.html
# returns a dictionary which uses the word token
# as the key and a list of 300 floats as the value
import io
import time

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    num_words, vec_size = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


#######################################################
# These 3 functions calculate the cosine similarity for
# any 2 word vectors


def dot_product(vec_a, vec_b):
    dot_prod = 0.0;
    for i in range(len(vec_a)):
        dot_prod += vec_a[i] * vec_b[i]
    return dot_prod

import math

def magnitude(vector):
    return math.sqrt(dot_product(vector, vector))

# The entry point function
def cosine_similarity(vec_a, vec_b):
    dot_prod = dot_product(vec_a, vec_b)
    magnitude_a = magnitude(vec_a)
    magnitude_b = magnitude(vec_b)
    return dot_prod / (magnitude_a * magnitude_b)


#loading the file
vectors = load_vectors("FastText100K.txt")

print("Start time:", time.strftime("%H:%M:%S"))

#input loop
while True:
    word = input("Enter a word (blank to quit): ").strip()
    if word == "":
        break
    if word not in vectors:
        print("Word not found in the dictionary")
        continue


    #compute the similarites
    entered_vec = vectors[word]
    similarities = []

    for w, vec in vectors.items():
        if w == word:
            continue
        score = cosine_similarity(entered_vec, vec)
        similarities.append((score, w))

    #get the top 5
    top5 = sorted(similarities, reverse=True)[:5]

    #print results
    print("The words with the highest cosine similarity are:")
    for score, w in top5:
        print(f"{score:.16f}  {w}")
    print()

print("Finish time:", time.strftime("%H:%M:%S"))
