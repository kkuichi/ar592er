import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import random
import torch
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW


# GPU
device = torch.device("cpu")

# Načítanie dátasetov pre trénovaciu, testovaciu a validačnú množinu
my_train = pd.read_csv("myds_train.csv")
my_test = pd.read_csv("myds_test.csv")
my_valid = pd.read_csv("myds_valid.csv")

# Sťahovanie potrebných jazykových zdrojov pre spracovanie textu
nltk.download('wordnet')        
nltk.download('omw-1.4')
nltk.download('stopwords')     
nltk.download('punkt_tab')  
stop_words = set(stopwords.words('english'))

# Funkcia na predspracovanie textu
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+|https?://\S+|www\.\S+", "", text)  
    text = re.sub(r"<.*?>", "", text)  
    text = re.sub(r"_\w+", "", text) 
    text = re.sub(r'\d+', '', text) 
    text = re.sub(r'[' + re.escape(string.punctuation) + r']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip() 
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w) for w in filtered_sentence]
    clean_tokens = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    return ' '.join(clean_tokens)


# Aplikácia funkcie `preprocess` na jednotlivé datasety
my_train['clean_text'] = my_train['text'].apply(preprocess)
my_test['clean_text'] = my_test['text'].apply(preprocess)
my_valid['clean_text'] = my_valid['text'].apply(preprocess)

# Načítanie predtrénovaného BERT modelu a tokenizéra
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


# Tokenizáci sekvencie v trénovacej sade
tokens_train = tokenizer.batch_encode_plus(
    my_train['clean_text'].tolist(),
    max_length = 25,
    padding='max_length',
    truncation=True
)

# Tokenizáci sekvencie v validačnej sade
tokens_val = tokenizer.batch_encode_plus(
    my_valid['clean_text'].tolist(),
    max_length = 25,
    padding='max_length',
    truncation=True
)

# Tokenizáci sekvencie v testovacej sade
tokens_test = tokenizer.batch_encode_plus(
    my_test['clean_text'].tolist(),
    max_length = 25,
    padding='max_length',
    truncation=True
)

# Konverzia výstupov tokenizácie na tenzory
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(my_train['label'].tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(my_valid['label'].tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(my_test['label'].tolist())

print(val_seq.shape)
print(val_mask.shape)
print(val_y.shape)



batch_size = 32

# Zabalenie tenzorov
train_data = TensorDataset(train_seq, train_mask, train_y)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_data = TensorDataset(val_seq, val_mask, val_y)
val_dataloader = DataLoader(val_data, batch_size=32)

# Zmrazenie všetkých parametrov BERT modelu
for param in bert.parameters():
    param.requires_grad = False


# Definícia triedy modelu s vlastnou architektúrou nad BERT

class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert 
        self.dropout = nn.Dropout(0.2)
        self.gelu = nn.GELU()
        self.fc1 = nn.Linear(768,512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_hs = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        x = self.fc1(cls_hs)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.softmax(x)

        return x


model = BERT_Arch(bert)
model = model.to(device)
# Definovanie optimalizátora 
optimizer = AdamW(model.parameters(),lr = 2e-5) 


# Výpočet váh tried
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(my_train['label']),
    y=my_train['label']
)

print("Class Weights:",class_weights)



# Konverzia váh do tenzora
weights= torch.tensor(class_weights,dtype=torch.float)
weights = weights.to(device)
cross_entropy  = nn.NLLLoss(weight=weights) 


epochs = 9
total_steps = len(train_dataloader) * epochs

# Nastavenie plánovača učenia s "warmup" fázou
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Funkcia, ktorá trénuje model
def train():
    
    model.train()
    total_loss = 0, 0
    total_correct = 0
  
    total_preds=[]
    total_labels = []
  
    for step,batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch 
        model.zero_grad()        
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)
        total_loss = total_loss + loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        preds = preds.detach().cpu().numpy()
        labels_cpu = labels.cpu().numpy()
        pred_labels = np.argmax(preds, axis=1)
        total_correct += np.sum(pred_labels == labels_cpu)

        total_preds.append(preds)
        total_labels.extend(labels_cpu)

    avg_loss = total_loss / len(train_dataloader)
    accuracy = total_correct / len(total_labels)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, accuracy, total_preds
  


# Funkcia na vyhodnotenie modelu
def evaluate():
    
    print("\nEvaluating...")
    model.eval()
    total_loss, total_correct = 0, 0
    total_preds = []
    total_labels = []

    for step,batch in enumerate(val_dataloader):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        with torch.no_grad():
            preds = model(sent_id, mask)
            loss = cross_entropy(preds,labels)
            total_loss = total_loss + loss.item()
            preds_cpu = preds.detach().cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            pred_labels = np.argmax(preds_cpu, axis=1)
            total_correct += np.sum(pred_labels == labels_cpu)

            total_preds.append(preds_cpu)
            total_labels.extend(labels_cpu)

    avg_loss = total_loss / len(val_dataloader)
    accuracy = total_correct / len(total_labels)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, accuracy, total_preds
   

# Nastavenie počiatočnej najlepšej straty na nekonečno
best_valid_loss = float('inf')

# Prázdne zoznamy pre uchovanie tréningovej a validačnej straty pre každú epochuh
train_losses=[]
valid_losses=[]
train_accuracies = []
valid_accuracies = []

# Pre každú epochu
for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    train_loss, train_acc, _ = train()
    valid_loss, valid_acc, _ = evaluate()
    
    # Uloženie najlepšieho modelu
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    
    # Uloženie strát
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accuracies.append(train_acc)
    valid_accuracies.append(valid_acc)
    
    print(f'\nTraining Loss: {train_loss:.3f}, Training Accuracy: {train_acc:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}, Validation Accuracy: {valid_acc:.3f}')


# Načítanie najlepších váh modelu
path = 'saved_weights.pt'
model.load_state_dict(torch.load(path))


with torch.no_grad():
    outputs = model(test_seq.to(device), test_mask.to(device))  
    probs = torch.exp(outputs) 
    preds = torch.argmax(probs, axis=1).cpu().numpy()  
    probs_class1 = probs[:, 1].cpu().numpy() 


# Výpočet metrík
accuracy = accuracy_score(test_y, preds)
precision = precision_score(test_y, preds, average='macro')  
recall = recall_score(test_y, preds, average='macro')
f1 = f1_score(test_y, preds, average='macro')

# Výpis výsledkov
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print(classification_report(test_y, preds))


# Krivka straty pre tréning a validáciu
epochs = range(len(train_losses))
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_losses, label='Straty trénovania')
plt.plot(epochs, valid_losses, label='Strata validácii')
plt.xlabel('Epochy')
plt.ylabel('Strata')
plt.title('Strata za epochu')
plt.legend()
plt.grid(True)
plt.show()

# Krivka úspešnosti
plt.figure()
plt.plot(epochs, train_accuracies, label='Úspešnosť trénovania')
plt.plot(epochs, valid_accuracies, label='Úspešnosť validácii')
plt.xlabel('Epochy')
plt.ylabel('Úspešnosť')
plt.title('Úspešnosť za epochu')
plt.legend()
plt.grid(True)
plt.show()

# Matica zámien
cm = confusion_matrix(test_y, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Toxic', 'Toxic'], yticklabels=['Non-Toxic', 'Toxic'])
plt.xlabel('Predikovaná')
plt.ylabel('Skutočná')
plt.title('Matica Zámen')
plt.show()

# ROC krivka
fpr, tpr, _ = roc_curve(test_y, probs_class1)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC krivka (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('Falošne Pozitívna')
plt.ylabel('Skutočne Pozitívna')
plt.title('ROC Krivka')
plt.legend()
plt.show()




# Nastavenie seed
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Načítanie konkrétneho príkladu z testovacieho datasetu
example_index = 47  
text = my_test.loc[example_index, "clean_text"]

set_seed(45)

model.eval()

# Tokenizácia vstupného textu
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
print(f"text: {text}")
print(f"inputs: {inputs}")
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].float().to(device)  

# Implementácia LRP metódy
def simple_lrp(model, input_ids, attention_mask):
    model.train() 
    model.zero_grad()
    model.bert.embeddings.word_embeddings.weight.requires_grad_(True)

    saved_embeddings = {}

    # Hook na zachytenie embeddingov počas forward prechodu
    def forward_hook(module, input, output):
        output.retain_grad() 
        saved_embeddings['embeddings'] = output

    hook = model.bert.embeddings.word_embeddings.register_forward_hook(forward_hook)

    output = model(input_ids, attention_mask)
    target_class = torch.argmax(output, dim=1)
    print(f"Predikovaná trieda: {target_class.item()}")
    realclass = my_test.loc[example_index, "label"]
    print(f"Skutočná trieda: {realclass}")

    output[0, target_class].backward()
    hook.remove()
    embeddings = saved_embeddings['embeddings']
    gradients = embeddings.grad
    relevance_scores = (embeddings * gradients).sum(dim=-1).squeeze()

    model.eval()  
    return relevance_scores.detach().cpu().numpy()


# Výpočet relevancie jednotlivých tokenov 
relevance = simple_lrp(model, input_ids, attention_mask)
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Normalizácia relevancie do rozsahu 0 až 1
relevance = (relevance - np.min(relevance)) / (np.max(relevance) - np.min(relevance) + 1e-10)

# Výpis tokenov a ich relevancie
for token, score in zip(tokens, relevance):   
    if token not in ['[CLS]', '[SEP]', '[PAD]']:
        print(f"{token}: {score:.4f}")

# Vizualizácia tokenov
def visualize_tokens(tokens, relevance_scores):
    html_output = "<div style='font-family:\"Times New Roman\", Times, serif; line-height:2;'>"
    for token, score in zip(tokens, relevance_scores):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            red = int(255 * (1 - score))
            blue = int(255 * score)
            color = f"rgba({red}, 0, {blue}, 0.6)"
            html_output += f"<span style='background-color: {color}; padding:2px 6px; margin:2px; border-radius: 4px; display:inline-block;'>{token}</span> "
    html_output += "</div>"
    return html_output

# Generovanie HTML výstupu
html_vis = visualize_tokens(tokens, relevance)

# Uloženie vizualizácie do HTML súboru
with open("lrp_visualization.html", "w", encoding="utf-8") as f:
    f.write(html_vis)

print("Hotovo! Vizualizácia uložená do súboru lrp_visualization.html")

