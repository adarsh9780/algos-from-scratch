import torch
from torch.nn.functional import one_hot as F

START_CHAR = "<S>"
EOW_CHAR = "<E>"

with open("names.txt", "r") as i:
    data = i.readlines()
    names = [word.strip() for word in data]


vocabulary = sorted(set("".join(names)))
char_to_index = {char: i + 1 for i, char in enumerate(vocabulary)}
char_to_index[START_CHAR] = 0
char_to_index[EOW_CHAR] = len(vocabulary) + 1

index_to_char = {v: k for k, v in char_to_index.items()}

N = torch.zeros((28, 28), dtype=torch.int32)

for name in names:
    name = [START_CHAR] + list(name) + [EOW_CHAR]
    for i in range(len(name) - 1):
        ch1, ch2 = name[i], name[i + 1]
        N[char_to_index[ch1], char_to_index[ch2]] += 1

P = N.float()
P /= P.sum(dim=1, keepdim=True)

idx = 0
word = ""
while True:
    p = P[idx]
    sampled_index = torch.multinomial(p, num_samples=1, replacement=True).item()

    if index_to_char[sampled_index] == EOW_CHAR:
        print(word)
        break

    word += index_to_char[sampled_index]
    idx = sampled_index


"""
1. What we need is a way to measure the performance of our model.
Which means we need a number to measure the performance

We will use something called Maximum Likelihood Estimation(MLE).
MLE:

Probability vs Likelihood
Suppose we toss a coin 10 times. 1 = Head, 0 = Tail
Probability: 1, 1, 0, 0, 1, 0, 1, 1, 1, 1

P(1) = 7/10 = 0.7
P(0) = 3/10 = 0.3

The probability of observing a data where 7 heads and 3 tails come is:
Likelihood (L) = P(1) * P(0)

0.7 * 0.3

as we know probabilities are numbers between 0-1. these are small numbers.
as we keep multipliying it, the final number (result) will become smaller
and smaller.

For convenience, we use something called Log Likelihood. it allows us to do
2 cool things.
i) it turns multiplication to addition because
log(a * b) = log(a) + log(b)
ii) with every new probability multiplied (added in case of log likelihood)
it increases the number.

One thing we have to keep on mind is that, log of values between 0-1 is a
negative number. L decreases with each probability being multiplied.

therefore, we make one more convenience, we reverse the sign, aka, Negative
Log Likelihood.

NLL = -[log(0.7) + log(0.3)] = 1.5606477

So, for our use case, we will check the NLL for our bigrams. and that would
become our loss function which we will minimize.

we will need all possible bigram pairs: which is N
take their probabilities: which is P
take their logs: P.log()
take the sum: P.log().sum()
"""

nll = P.log().sum()  # this would give nan, why?

# because the last row, which represents the combination of <S> and <E>
# never happens. start and end char never occurs at the same time.
# so those counts are 0, probabilities for them is undefined, thus nan
# log of nan is nan
# so we will only keep P except the last row

Pnew = P[0:26, :]

nll = Pnew.log().sum()

# there is another problem here, this can be infinity if any of the number
# is 0, because log(0) = inf
# therefore, we will add 1 to each element

nll = (Pnew + 1).log().sum()

# so we got a number, but even this is wrong. we are calculating NLL for
# all possible combination (bigrams) and some of which do not even occur in
# the english language. so we have to calculate NLL for only those bigrams
# which occur i.e. all the names in names.txt file.

# second, we are adding 1 to each probability which is wrong.
# we will add 1 to each count. to N, not to P

P = (N + 1).float()
P /= P.sum(dim=1, keepdim=True)

nll = 0.0
n = 0
for name in names:
    name = [START_CHAR] + list(name) + [EOW_CHAR]
    for i in range(len(name) - 1):
        ch1, ch2 = name[i], name[i + 1]
        nll -= P[char_to_index[ch1], char_to_index[ch2]].log()
        n += 1

average_nll = nll / n

"""
Why is the baseline random loss -log(1/28) ≈ 3.33 instead of 1/(28 C 2) or 1/(28 * 28)?

1. The Game We Are Playing: "Guess the NEXT character" (Conditional Probability)
   A bigram language model does NOT guess both characters from scratch at the same time.
   The first character is already known and given to the model as input.

   We are only asking:
       "Given that the current character is 'm', what is the ONE next character?"

   There are exactly 28 possibilities for that next character (26 letters + <S> + <E>).
   If a model knows nothing and guesses completely at random:
       P(correct) = 1 / 28
       Loss per step = -log(1 / 28) ≈ 3.33

   Because `average_nll = nll / n` divides the total loss by the number of single-character
   prediction steps, the average loss for a pure random guesser is ≈ 3.33.

2. What about all possible bigrams? (Combinatorics vs. Joint Probability)
   - Why not 28 C 2 (378)?
     Combinations (28 C 2) are for UNORDERED pairs WITHOUT repetition (e.g. {a, b}).
     Language bigrams are ORDERED and ALLOW REPETITION ('th' != 'ht', and 'mm' in 'emma' is valid).
     The total possible ordered bigrams in a 28x28 grid is 28 * 28 = 784.

   - When would the baseline be 1 / 784?
     Only if you had to guess BOTH the first and second letter simultaneously out of thin air:
         P(ch1, ch2) = P(ch1) * P(ch2) = (1/28) * (1/28) = 1 / 784
         Loss = -log(1 / 784) = -log(1/28) + -log(1/28) ≈ 3.33 + 3.33 = 6.66

   Because the first character is always handed to us as input, we evaluate each step as
   choosing 1 out of 28 possibilities: P(next | current).
"""

# Doing the same thing as above using a Neural Network
# we will see that after training the model, the NN will converege to the same
# probabilities as the one we calculated above.

# step 1: create a training set for nn
# for each word in the vocabulary, we will iterate through each char
# for character i, character at i+1 is y

xs, ys = [], []
for name in names:
    name = [START_CHAR] + list(name) + [EOW_CHAR]
    for i in range(len(name) - 1):
        ch1, ch2 = name[i], name[i + 1]
        xs.append(char_to_index[ch1])
        ys.append(char_to_index[ch2])

Xs = torch.tensor(xs)
Ys = torch.tensor(ys)

Xenc = F(Xs, num_classes=28).float()
Yenc = F(Ys, num_classes=28).float()

# let's construct first neuron
W = torch.randn((28, 28), dtype=torch.float32, requires_grad=True)

# because we will use softmax later on as loss function
# it is a convetion to call the output of layer prior to
# softmax layer as logits, so logits are fed to softmax
# and softmax gives us probabilities
learning_rate = -10.0
step = 10
for k in range(100):
    # FORWARD PASS
    # Step 1: calculate logits
    logits = Xenc @ W
    # Step 2: do an exponential
    counts = logits.exp()
    # Step 3: Normalize it
    probs = counts / counts.sum(dim=1, keepdim=True)
    # Step 2 + Step 3 = SOFTMAX

    # Calculate loss
    # torch.arange(5) = tensor([0, 1, 2, 3, 4, 5])
    # [0, 5], [1, 13] = for zeroth row, we want 5 element
    # suppose we only have 6 rows or bigrams
    # probs[0, 5]
    target_probs = probs[torch.arange(len(xs)), ys]
    loss = -target_probs.log().mean() + 0.01 * (W**2).mean()

    if step % 10 == 0:
        print(loss)

    W.grad = None
    loss.backward()
    W.data += learning_rate * W.grad
    step += 1
