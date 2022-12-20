# %%


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# TODO make dataset


class TextDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.corpus = "The wide road shimmered in the hot sun".lower().split()
        unique_words = set(self.corpus)
        self.idx_to_token = {idx: token for idx,
                             token in enumerate(unique_words)}
        self.token_to_idx = {token: idx for idx,
                             token in self.idx_to_token.items()}
        window_start_idx = 0
        window_size = 3
        assert window_size % 2 == 1
        self.skipgrams = []
        while window_start_idx + window_size < len(self.corpus):
            window = self.corpus[window_start_idx:window_start_idx+window_size]
            central_word = window.pop(int((window_size-1)/2))
            skip_grams_for_this_window = []
            for context_word in window:
                skip_grams_for_this_window.append((central_word, context_word))
            self.skipgrams.extend(skip_grams_for_this_window)
            window_start_idx += 1
        print(self.skipgrams)

    def __len__(self):
        return len(self.skipgrams)

    def __getitem__(self, idx):
        central_word, context_word = self.skipgrams[idx]
        central_word_idx = self.token_to_idx[central_word]
        context_word_idx = self.token_to_idx[context_word]
        return central_word_idx, context_word_idx

# DEFINE WORD2VEC


class Word2Vec(torch.nn.Module):
    def __init__(self, n_words, embedding_dim):
        super().__init__()
        self.central_word_embeddings = torch.nn.Embedding(
            n_words, embedding_dim)
        self.context_word_embeddings = torch.nn.Linear(
            embedding_dim, n_words, bias=False)

    def forward(self, X):
        print(X)
        central_word_embedding = self.central_word_embeddings(X)
        similarities = self.context_word_embeddings(central_word_embedding)

        # cwe = self.context_word_embeddings.parameters()
        # similarities = torch.matmul(cwe.T, central_word_embedding)

        return similarities
        # probability_dist_over_context_words = F.softmax(similarities)
        # return probability_dist_over_context_words


def train(model, dataloader, epochs=10):
    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for batch in dataloader:
            word_indices_of_central_word_for_each_example_in_batch, labels = batch
            similarities = model(
                word_indices_of_central_word_for_each_example_in_batch)  # predict
            loss = F.cross_entropy(similarities, labels)  # calculat loss
            loss.backward()  # calculates gradients by doing backprop
            optimiser.step()  # take optimisation step
            optimiser.zero_grad()  # zero grad
            print(loss.item())


dataset = TextDataset()
train_len = round(0.8*len(dataset))
val_len = round(0.1*len(dataset))
test_len = len(dataset) - val_len - train_len
lengths = [train_len, val_len, test_len]
train_set, val_set, test_set = random_split(dataset, lengths)
# dataloader shuffles and batches the dataset
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)

word2vec = Word2Vec(n_words=len(dataset.idx_to_token), embedding_dim=128)

train(word2vec, train_loader)

# %%
