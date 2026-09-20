from pathlib import Path

import torch
from torch.nn.functional import cross_entropy, softmax

START_CHAR = "<S>"
END_CHAR = "<E>"


def read_names(filepath: str | Path) -> list[str]:
    with open(filepath, "r") as i:
        queue = i.readlines()
        names = [q.strip() for q in queue]

    return names


def create_vocabulary(names: list[str]) -> list[str]:
    vocabulary = sorted({n for name in names for n in list(name)})
    vocabulary = [START_CHAR] + list(vocabulary) + [END_CHAR]

    return vocabulary


def train_test_split(
    names: list[str],
    train_size: float = 0.8,
    test_size: float = 0.1,
    val_size: float = 0.1,
) -> tuple(list[str]):

    if train_size + test_size + val_size != 1.0:
        raise ValueError("train, test, and val should add up to 1.0")

    import random

    random.shuffle(names)
    n1 = int(0.8 * len(names))
    n2 = int(0.9 * len(names))

    train_names = names[:n1]
    dev_names = names[n1:n2]
    test_names = names[n2:]

    return train_names, dev_names, test_names


def build_dataset(
    start_char: str, end_char: str, names: list[str], block_size: int = 3
):
    block_size = 3
    x, y = [], []
    for w in names:
        context = [0] * block_size
        w = [start_char] + list(w) + [end_char]

        for ch in w[1:]:
            ix = stoi[ch]
            x.append(context)
            y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(x)
    Y = torch.tensor(y)

    return X, Y


if __name__ == "__main__":
    names = read_names("names.txt")
    vocabulary = create_vocabulary(names)

    stoi = {char: i for i, char in enumerate(vocabulary)}
    itos = {index: char for char, index in stoi.items()}

    train_names, dev_names, test_names = train_test_split(names)
    Xtr, Ytr = build_dataset(
        start_char=START_CHAR, end_char=END_CHAR, names=train_names
    )
    Xde, Yde = build_dataset(start_char=START_CHAR, end_char=END_CHAR, names=dev_names)
    Xte, Yte = build_dataset(start_char=START_CHAR, end_char=END_CHAR, names=test_names)

    C = torch.randn((28, 10), dtype=torch.float32, requires_grad=True)

    # first hidden layer, 100 neurons
    W1 = torch.randn((30, 200), dtype=torch.float32, requires_grad=True)
    b1 = torch.randn(200, dtype=torch.float32, requires_grad=True)

    W2 = torch.randn((200, 28), dtype=torch.float32, requires_grad=True)
    b2 = torch.randn(28, dtype=torch.float32, requires_grad=True)

    parameters = [C, W1, W2, b1, b2]
    epochs = 50_000
    learning_rate = 0.01

    for count, i in enumerate(range(epochs)):
        # only calculate gradient for a batch and use that
        # this will make it much faster
        ix = torch.randint(0, Xtr.shape[0], (128,))
        embeddings = C[Xtr[ix]]

        h1 = ((embeddings.view(-1, 30) @ W1) + b1).tanh()

        logits = (h1 @ W2) + b2
        # it does the same thing as h3 = h2.exp()
        # prob = h3/h3.sum(dim=1, keepdim=True)
        # loss = prob[torch.arange(Y), Y].log().mean()
        loss = cross_entropy(logits, Ytr[ix])

        for p in parameters:
            p.grad = None

        loss.backward()

        # if count % 100 == 0:
        #     print(loss.item())

        for p in parameters:
            p.data += p.grad * -learning_rate

    # calculate overall loss on training
    emb = C[Xtr]
    h1 = emb.view(-1, 30) @ W1 + b1
    logits = h1.tanh() @ W2 + b2
    loss = cross_entropy(logits, Ytr)
    print(f"Training Loss: {loss}")

    # calculate overall loss on dev
    emb = C[Xde]
    h1 = emb.view(-1, 30) @ W1 + b1
    logits = h1.tanh() @ W2 + b2
    loss = cross_entropy(logits, Yde)
    print(f"Dev Loss: {loss}")

    # calculate overall loss on test: only one time
    emb = C[Xte]
    h1 = emb.view(-1, 30) @ W1 + b1
    logits = h1.tanh() @ W2 + b2
    loss = cross_entropy(logits, Yte)
    print(f"Dev Loss: {loss}")

    # Sampling from these or generating names using this model
    context = [0] * 3
    word = ""
    while True:
        emb = C[torch.tensor([context])]
        h1 = emb.view(-1, 30) @ W1 + b1
        logits = h1.tanh() @ W2 + b2
        probs = softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        char = itos[ix]

        if char == END_CHAR:
            print(word)
            break

        word += char
        context = context[1:] + [ix]
