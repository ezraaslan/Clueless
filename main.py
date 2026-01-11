import gensim.downloader as api
from numpy import dot
from numpy.linalg import norm

model = api.load("glove-wiki-gigaword-50")
print("Model loaded.")

def similarity(word1, word2):
    if word1 not in model or word2 not in model:
        return 0.0
    return dot(model[word1], model[word2]) / (norm(model[word1]) * norm(model[word2])) * 100

TARGET = "volcano"

print("Guess the word!")

while True:
    guess = input("> ").strip().lower()
    if guess == "quit":
        break
    elif guess == TARGET:
        print("You got it!")
        break

    score = similarity(guess, TARGET)
    print(f"Similarity: {round(score, 1)}%")

