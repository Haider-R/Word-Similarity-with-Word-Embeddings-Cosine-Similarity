import numpy as np
import sys
import io
from datetime import datetime

#add limit to how many vectors you can add because my laptop has a hard time loading the entire wiki-news-300d-1M file
def load_vectors(fname, limit=None):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    count = 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        vec = np.array(list(map(float, tokens[1:])))
        vec /= np.linalg.norm(vec)
        data[tokens[0]] = vec
        count += 1
        if limit and count >= limit:
            break
    return data



def find_best_analogies(vectors, w1, w2, w3, top_n=20):
    exclude = {w1, w2, w3}
    v1, v2, v3 = vectors[w1], vectors[w2], vectors[w3]

    #analogy vector (normalize after operation)
    target = v2 - v1 + v3
    target /= np.linalg.norm(target)

    sims = []
    for word, vec in vectors.items():
        if word in exclude:
            continue
        cosine = float(np.dot(target, vec))
        sims.append((cosine, word))

    sims.sort(reverse=True, key=lambda x: x[0])
    return sims[:top_n]


def main():
    print("Lab #3 - by Haider Rizvi")
    print("Analogies take the form: A is to B as C is to D")
    print("Example: 'man is to woman as king is to queen'\n")

    filename = "FastText100K.txt"

    #uncomment below to use the other file
    #filename = "wiki-news-300d-1M.txt"
    
    print("Loading word vector dictionary")
    print(datetime.now().strftime("%H:%M:%S"))
    vectors = load_vectors(filename, limit=500000) ##change the limit to however many files you want to be loaded
    print(datetime.now().strftime("%H:%M:%S"))
    print("Word vector dictionary is loaded\n")



    while True:
        user_input = input("\nEnter 3 words separated by spaces (or press Enter to quit): ").strip().lower()
        if not user_input:
            print("Goodbye.")
            sys.exit(0)

        words = user_input.split()
        if len(words) != 3:
            print("Please enter exactly three words.")
            continue

        #use the loaded 'vectors' (do NOT reassign it)
        missing = [w for w in words if w not in vectors]
        if missing:
            print("Word(s) not found:", ", ".join(missing))
            continue

        w1, w2, w3 = words
        print(f"\nAnalogy: {w1} is to {w2} as {w3} is to ____\n")
        results = find_best_analogies(vectors, w1, w2, w3, 20)

        for score, word in results:
            print(f"{score:.4f}\t{word}")

        print("\n--- Try another or press Enter to quit ---")

main()
