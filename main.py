import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm


class ExpertLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

class MoELayer(nn.Module):
    def __init__(self, input_size, output_size, num_experts, k):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.k = k

        self.experts = nn.ModuleList([ExpertLayer(input_size, output_size) for _ in range(num_experts)])
        self.gate = nn.Linear(input_size, num_experts)
        self.expert_biases = nn.Parameter(torch.zeros(num_experts))

    def forward(self, x):
        # Gate computation
        gate_logits = self.gate(x) + self.expert_biases
        gate_probs = torch.sigmoid(gate_logits)
        
        # Top-k gating
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts])
        
        # Gather and combine expert outputs
        batch_size, seq_len, _ = x.shape
        indices = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.output_size)
        expert_outputs = expert_outputs.gather(0, indices.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)
        
        final_output = (expert_outputs * top_k_probs.unsqueeze(-1)).sum(dim=-2)
        
        return final_output, top_k_indices

class ToyMoEModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_experts, k):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.moe_layer = MoELayer(embed_size, hidden_size, num_experts, k)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, expert_indices = self.moe_layer(x)
        x = self.output_layer(x)
        return x, expert_indices
    
class TextDataset(Dataset):
    def __init__(self, encoded_texts, max_length):
        self.encoded_texts = encoded_texts
        self.max_length = max_length

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        item = self.encoded_texts[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'][:self.max_length]),
            'attention_mask': torch.tensor(item['attention_mask'][:self.max_length])
        }

def load_and_preprocess_data(max_length=128):
    #dataset = load_dataset("tiny_shakespeare")
    dataset = load_dataset("wikitext", "wikitext-2-v1")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    train_dataset = TextDataset(tokenized_datasets["train"], max_length)
    
    return train_dataset, tokenizer
    
def calculate_maxvio(expert_counts):
    avg_count = expert_counts.float().mean()
    max_violation = torch.max(torch.abs(expert_counts.float() - avg_count) / avg_count)
    return max_violation.item()

def plot_metrics(losses, maxvios):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(maxvios)
    plt.title('MaxVio')
    plt.xlabel('Step')
    plt.ylabel('MaxVio')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def train(model, train_dataset, tokenizer, num_epochs=3, batch_size=32, learning_rate=1e-3, update_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    maxvios = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        expert_counts = torch.zeros(model.moe_layer.num_experts).to(device)
        
        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            optimizer.zero_grad()
            
            outputs, expert_indices = model(input_ids)
            
            # Shift the targets for next token prediction
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            losses.append(loss.item())
            
            # Count expert usage
            batch_expert_counts = torch.bincount(expert_indices.flatten(), minlength=model.moe_layer.num_experts)
            expert_counts += batch_expert_counts
            
            # Calculate and store MaxVio
            maxvio = calculate_maxvio(batch_expert_counts)
            maxvios.append(maxvio)
            
            # Update expert biases
            avg_count = expert_counts.float().mean()
            for i, count in enumerate(expert_counts):
                error = count.float() - avg_count
                model.moe_layer.expert_biases.data[i] += update_rate * torch.sign(error)
            
            expert_counts.zero_()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Plot metrics after training
    plot_metrics(losses, maxvios)


def main():
    vocab_size = 30522  # BERT vocab size
    embed_size = 256
    hidden_size = 512
    num_experts = 8
    k = 2

    train_dataset, tokenizer = load_and_preprocess_data()
    
    model = ToyMoEModel(vocab_size, embed_size, hidden_size, num_experts, k)
    
    train(model, train_dataset, tokenizer)

if __name__ == "__main__":
    main()
