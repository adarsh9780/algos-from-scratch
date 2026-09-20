import torch
from torch.nn.functional import one_hot

with open("names.txt", "r") as i:
    queue = i.readlines()
    names = [q.strip() for q in queue]

vocabulary = sorted({n for name in names for n in list(name)})
vocabulary = ["<S>"] + list(vocabulary) + ["<E>"]

stoi = {char: i for i, char in enumerate(vocabulary)}
itos = {index: char for char, index in stoi.items()}

print(names[0:10])

# create a matrix to store the bigram count
# bigram = 2 characters at the same time
# we need to know the count of pair of character, given the one

# notice how we are giving default count = 1 to each bigram
N = torch.ones((28, 28), dtype=torch.float32)

total_bigram = 0
bigrams = []
for name in names:
    name = ["<S>"] + list(name) + ["<E>"]
    for i, j in zip(name, name[1:]):
        id1, id2 = stoi[i], stoi[j]
        bigrams.append((i, j))
        N[id1, id2] += 1
        total_bigram += 1

# convert the counts to probabalities
P = N / N.sum(dim=1, keepdim=True)

# also notice that how these probabilities are based on the actual
# data we have, so we are building a mathematical model

# we need to measure the performance of model, and since we are
# calculating probabilities, we will use Maximum Likelihood estimation

# to put in simple terms, MLE = Probability of all the events happening
# together

# example: "ad" is a bigram, MLE = P(d|a) => probability of d coming
# just after "a"

# and since probabilities are numbers between 0-1, the result will be
# a small number, to avoid which we use log probabilities

# taking log, has two advantages:
# 1) it turns multiplication to addition problem
# 2) with each additional terms, the sum increases, which intuitively
# we know we need to minimize
# but log of any number between 0-1 will be negative, we take negative log
# and we only want to take the logs of those bigrams which actually
# exist in our data
nll = 0.0
for bi in bigrams:
    char1, char2 = bi[0], bi[1]
    id1, id2 = stoi[char1], stoi[char2]
    nll += -P[id1, id2].log()

# now each number in the matrix P represents the Probabilities

average_nll = nll / total_bigram
print(average_nll)

# so the mathematical model's average loss is 2.4543. lower the better
# Assume that each number in the bigram has equal chance
# of occurence. in that case, every number in N would have been 1/28

B = torch.full((28, 28), 1 / 28, dtype=torch.float32)
bnll = 0.0
for bi in bigrams:
    char1, char2 = bi[0], bi[1]
    id1, id2 = stoi[char1], stoi[char2]
    bnll += -B[id1, id2].log()

average_baseline_nll = bnll / total_bigram
print(average_baseline_nll)

# so if each character has equal chance of occuring
# then our loss is 3.3296 compared to 2.4543 which
# our mathematical model learnt from our data
# so we are already performing better

# how to take samples from the mathematical mode
# we will use something called Multinomial
# Multinomial takes a weight matrix, and produces
# desired number of samples based on the weight
start_index = 0
word = ""
while True:
    id = torch.multinomial(P[start_index], num_samples=1, replacement=True)
    char = itos[id.item()]
    if char == "<E>":
        print(word)
        break
    word += char
    start_index = id.item()

# no we will do the same thing using a single neuron
# and we will see that our neural network will converge to the same
# probabilities which our mathematical model learnt

# we need to think how we would pass the input to this neuron
# we will give it a vector of length 28
# each nuber in the vector represents of that character is present or not
# this is called one hot encoding

# for the very first word "<S>"
# one hot encoding is [1, 0, 0, ... 25 more times]
# one hot encoding of "a": [0, 1, 0, ... 25 more times]

# start with random weights between (-inf, inf)
W = torch.randn((28, 28), dtype=torch.float32, requires_grad=True)

# before we can use one hot encoding, we need to prepare traning
xs = []
ys = []
for bi in bigrams:
    char1, char2 = bi[0], bi[1]
    id1, id2 = stoi[char1], stoi[char2]
    xs.append(id1)
    ys.append(id2)

Xs = torch.tensor(xs)
Ys = torch.tensor(ys)

learning_rate = -100.0
epochs = 100

Xenc = one_hot(Xs, num_classes=28).float()
Yenc = one_hot(Ys, num_classes=28).float()
for count, i in enumerate(range(epochs)):
    logits = Xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(dim=1, keepdim=True)

    loss = -probs[torch.arange(len(Ys)), Ys].log().mean()

    if count % 10 == 0:
        print(loss)

    W.grad = None
    loss.backward()

    W.data += learning_rate * W.grad
