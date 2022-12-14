# %%


import torch

# TODO make dataset


class TextDataset(torch.utils.data.Dataset):
    def __init__(self):

        corpus = "The wide road shimmered in the hot sun".lower().split()
        window_start_idx = 0
        window_size = 3
        assert window_size % 2 == 1
        self.skipgrams = []
        while window_start_idx + window_size < len(corpus):
            window = corpus[window_start_idx:window_start_idx+window_size]
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
        return self.skipgrams[idx]
# TODO make dataloader


class SkipGram(torch.nn.Module):
    def __init__(self, n_words, embedding_dim):
        super().__init__()
        self.embeddings = torch.nn.Embedding(n_words, embedding_dim)
        self.decoder = torch.nn.Linear(embedding_dim, n_words)

    def forward(self, one_hot_central_word):
        embedding = self.embeddings(one_hot_central_word)
        logits_predicted_word = self.decoder(embedding)
        return logits_predicted_word


def train(model, dataloader, epochs=10):
    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for batch in dataloader:
            central_word, context_word = batch
            print(central_word)
            print(context_word)
            predicted_context_word = skipgram(central_word)
            loss = torch.nn.functional.cross_entropy(
                predicted_context_word, context_word)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            print("Loss:", loss.item())


dataset = TextDataset()
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=4)
skipgram = SkipGram(n_words=1000, embedding_dim=16)
train(skipgram, dataloader)

# %%
