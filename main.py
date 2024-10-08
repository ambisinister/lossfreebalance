import numpy as np
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
    def __init__(self, input_size, output_size, num_experts, k, use_aux_loss=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.k = k
        self.use_aux_loss = use_aux_loss

        self.experts = nn.ModuleList([ExpertLayer(input_size, output_size) for _ in range(num_experts)])
        self.gate = nn.Linear(input_size, num_experts)
        
        if not use_aux_loss:
            self.expert_biases = nn.Parameter(torch.zeros(num_experts))

    def forward(self, x):
        # s_{i,t}
        gate_output = self.gate(x)
        # use sigmoid gate instead of softmax
        gate_probs = torch.sigmoid(gate_output)

        # do top k based on s_{i,t} + b_i
        if not self.use_aux_loss:
            gate_logits = gate_output + self.expert_biases
        else:
            gate_logits = gate_output
            
        _, top_k_indices = torch.topk(gate_logits, self.k, dim=-1)

        # ...but make sure we use the unbiased s_{i,t} as the gate value
        top_k_probs = gate_probs.gather(-1, top_k_indices)

        # normalize to sum to 1
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # get the routed expert outputs
        batch_size, seq_len, _ = x.shape
        expert_outputs = torch.stack([expert(x) for expert in self.experts])
        indices = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.output_size)
        expert_outputs = expert_outputs.gather(0, indices.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)
        
        final_output = (expert_outputs * top_k_probs.unsqueeze(-1)).sum(dim=-2)

        return final_output, gate_probs, top_k_indices

class ToyMoEModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_experts, k, use_aux_loss=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.moe_layer = MoELayer(embed_size, hidden_size, num_experts, k, use_aux_loss)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.use_aux_loss = use_aux_loss

    def forward(self, x):
        x = self.embedding(x)
        x, gate_output, topk_idx = self.moe_layer(x)
        x = self.output_layer(x)
        return x, gate_output, topk_idx

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
    dataset = load_dataset("wikitext", "wikitext-2-v1")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    train_dataset = TextDataset(tokenized_datasets["train"], max_length)
    
    return train_dataset, tokenizer

# placeholder for visibility
def calculate_maxvio(expert_counts):
    avg_count = expert_counts.float().mean()
    #max_violation = torch.max(torch.abs(expert_counts.float() - avg_count) / avg_count)

    min_violation = torch.min(expert_counts.float()) / avg_count 
    max_violation = torch.max(expert_counts.float()) / avg_count 
    return [min_violation.item(), max_violation.item()]

def plot_metrics(losses, maxvios, use_aux_loss):
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')

    mins = [x[0] for x in maxvios]
    maxes = [x[1] for x in maxvios]
    window_size = 50
    
    plt.subplot(1, 3, 2)
    moving_avg = np.convolve(mins, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(maxvios)), moving_avg)
    plt.title('Tokens used by least-used expert (moving avg)')
    plt.xlabel('Step')
    plt.ylabel('Proportion of Average (1.0 is perfect balance)')

    plt.subplot(1, 3, 3)
    moving_avg = np.convolve(maxes, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(maxvios)), moving_avg)
    plt.title('Tokens used by most-used expert (moving avg)')
    plt.xlabel('Step')
    plt.ylabel('Proportion of Average (1.0 is perfect balance)')
    
    plt.tight_layout()
    plt.savefig(f'training_metrics_{"aux_loss" if use_aux_loss else "original"}.png')
    plt.close()

def calculate_auxiliary_loss(gate_probs):
    f_i = gate_probs.sum(dim=[0, 1]) / (gate_probs.size(0) * gate_probs.size(1))
    P_i = gate_probs.mean(dim=[0, 1])
    aux_loss = torch.sum(f_i * P_i)
    return aux_loss

def train(model, train_dataset, tokenizer, use_aux_loss=False, num_epochs=1, batch_size=32, learning_rate=1e-3, update_rate=1e-5, aux_loss_coeff=0.01):
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
        
        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            optimizer.zero_grad()
            
            outputs, gate_output, topk_idx = model(input_ids)
            
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            main_loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            if use_aux_loss:
                aux_loss = aux_loss_coeff * calculate_auxiliary_loss(gate_output)
                loss = main_loss + aux_loss
            else:
                loss = main_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            losses.append(loss.item())

            # maxvio to quantify distance from ideal load
            expert_counts = torch.bincount(topk_idx.flatten(),
                                           minlength=model.moe_layer.num_experts)            
            maxvio = calculate_maxvio(expert_counts)
            maxvios.append(maxvio)
            print(np.mean([x[0] for x in maxvios]), np.mean([x[1] for x in maxvios]))

            # adjust biases for experts
            if not use_aux_loss:
                avg_count = expert_counts.float().mean()
                for i, count in enumerate(expert_counts):
                    # b_i = b_i + u + sign(e_i)
                    # note: this is \bar{c_i} - c_i, NOT c_i - \bar{c_i}, which will push the network to
                    # be maximally unbalanced. Really important to get this part right!!!
                    error = avg_count - count.float()
                    model.moe_layer.expert_biases.data[i] += update_rate * torch.sign(error)
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    plot_metrics(losses, maxvios, use_aux_loss)

def main(use_aux_loss=False):
    vocab_size = 30522  # BERT vocab size
    embed_size = 256
    hidden_size = 512
    num_experts = 8
    k = 2

    train_dataset, tokenizer = load_and_preprocess_data()
    
    model = ToyMoEModel(vocab_size, embed_size, hidden_size, num_experts, k, use_aux_loss)
    
    train(model, train_dataset, tokenizer, use_aux_loss=use_aux_loss)

if __name__ == "__main__":
    main(use_aux_loss=False)
