import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import innvestigate #type:ignore
import shap
from wordcloud import WordCloud  #type:ignore
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.models import Sequential, Model# type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.layers import Embedding, Conv1D, Flatten, LSTM, Dense, Dropout, BatchNormalization # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.utils import resample


# Načítanie dátasetov pre trénovaciu, testovaciu a validačnú množinu
my_train = pd.read_csv("../dataset/hugging face/myds_train.csv")
my_test = pd.read_csv("../dataset/hugging face/myds_test.csv")
my_valid = pd.read_csv("../dataset/hugging face/myds_valid.csv")

# Sťahovanie potrebných jazykových zdrojov pre spracovanie textu
nltk.download('wordnet')        # Pre lemmatizáciu
nltk.download('omw-1.4')
nltk.download('stopwords')      # Anglické stop slová
nltk.download('punkt_tab')      # Pre tokenizáciu
# Nastavenie množiny stop slov
stop_words = set(stopwords.words('english'))   

# Nastavenie seed
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Funkcia na predspracovanie textu
def preprocess(text):
    text = text.lower()                                                         # Konverzia textu na malé písmená
    text = re.sub(r"http\S+|www\S+|https\S+|https?://\S+|www\.\S+", "", text)   # Odstránenie URL
    text = re.sub(r"<.*?>", "", text)                                           # Odstránenie HTML tagov
    text = re.sub(r"_\w+", "", text)                                            # Odstránenie používateľských mien (_meno alebo _Meno)
    text = re.sub(r'\d+', '', text)                                             # Odstránenie čísel
    text = re.sub(r'[' + re.escape(string.punctuation) + r']', ' ', text)       # Odstránenie interpunkcie
    text = re.sub(r'\s+', ' ', text).strip()                                    # Odstránenie nadbytočných medzier
    word_tokens = word_tokenize(text)                                           # Tokenizacia
    filtered_sentence = [w for w in word_tokens if w not in stop_words]         # Odstránenie stop slov
    lemmatizer = WordNetLemmatizer()                                            # Lematizacia
    lemmed = [lemmatizer.lemmatize(w) for w in filtered_sentence] 
    clean_tokens = [lemmatizer.lemmatize(w, pos='v') for w in lemmed]
    return clean_tokens


# Kontrola chýbajúcich hodnôt 
missing_values = my_train.isnull().sum()
print("Chýbajúce hodnoty v datasete:\n", missing_values)
print("Unikátne hodnoty sentimentu:", my_train['label'].unique())

# Distribúcia sentimentov v datasete
plt.figure(figsize=(6, 4))
sns.countplot(x="label", data=my_train, hue='label', legend=False, palette='Blues')
plt.xlabel("Label", fontsize=12)
plt.ylabel("Počet tweetov", fontsize=12)
plt.title('Distribúcia sentimentov v datasete', fontsize=14)
plt.xticks(ticks=[0, 1], labels=['Negatívny (0)', 'Pozitívny (1)'])
plt.show()

# Prieskum dĺžky textových príspevkov
my_train['text_length'] = my_train['text'].apply(len)

# Histogram rozloženia dĺžok tweetov
plt.hist(my_train['text_length'], bins=30, color='skyblue', edgecolor='darkblue')
plt.xlabel('Dĺžka tweetu', fontsize=10)
plt.ylabel('Počet', fontsize=10)
plt.title('Distribúcia dĺžky tweetov', fontsize=12)
plt.show()

# Výpočet priemernej dĺžky tweetov
average_length = my_train['text_length'].mean()
average_label_length = my_train.groupby('label')['text_length'].mean()
print("Priemerna hodnota dlzky: ", average_length)
print("Priemerne hodnoty dlzky podla label:\n", average_label_length)
# Výpočet počtu slov v jednotlivých tweetoch
my_train['word_count'] = my_train['text'].apply(lambda x: len(str(x).split()))

# Histogram počtu slov pre celý dataset
plt.figure(figsize=(7, 5))
plt.hist(my_train['word_count'], bins=30, color='plum', edgecolor='darkorchid')
plt.xlabel('Počet slov v tweete', fontsize=10)
plt.ylabel('Počet tweetov', fontsize=10)
plt.title('Histogram počtu slov v tweetoch', fontsize=12)
plt.show()

# Histogram počtu slov pre negatívne tweety (label 0)
plt.figure(figsize=(7, 5))
plt.hist(my_train[my_train['label'] == 0]['word_count'], bins=30, color='lightsteelblue', edgecolor='indigo', alpha=0.7)
plt.xlabel('Počet slov v tweete', fontsize=10)
plt.ylabel('Počet tweetov', fontsize=10)
plt.title('Histogram počtu slov v tweetoch pre triedu Negative', fontsize=12)
plt.show()

# Histogram počtu slov pre pozitívne tweety (label 1)
plt.figure(figsize=(7, 5))
plt.hist(my_train[my_train['label'] == 1]['word_count'], bins=30, color='lightsteelblue', edgecolor='indigo', alpha=0.7)
plt.xlabel('Počet slov v tweete', fontsize=10)
plt.ylabel('Počet tweetov', fontsize=10)
plt.title('Histogram počtu slov v tweetoch pre triedu Positive', fontsize=12)
plt.show()

# Priemerný počet slov pre celý dataset
average_word = my_train['word_count'].mean()
print("Priemerny pocet slov :\n", average_word)
# Priemerný počet slov podľa sentimentu
average_word_label = my_train.groupby('label')['word_count'].mean()
print("Priemerný počet slov podľa label:\n", average_word_label)

# orovnanie priemerného počtu slov medzi triedami
plt.figure(figsize=(6, 4))
average_word_label.plot(kind='bar', color=['skyblue', 'lightcoral'], edgecolor='gray')
plt.xlabel('Label', fontsize=10)
plt.ylabel('Priemerný počet slov v tweete', fontsize=10)
plt.title('Priemerný počet slov v tweetoch pre jednotlivé triedy', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['Negatívny (0)', 'Pozitívny (1)'], rotation=0)
plt.show()


# Spojenie všetkých tweetov do jedného textu
all_text = ' '.join(my_train['text']).lower()
# Vytvorenie WordCloud z celého datasetu
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='rocket').generate(all_text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Najpoužívanejšie slová v tweetoch', fontsize=12)
plt.show()

# Aplikácia funkcie `preprocess` na jednotlivé datasety
my_train['clean_text'] = my_train['text'].apply(preprocess)
my_test['clean_text'] = my_test['text'].apply(preprocess)
my_valid['clean_text'] = my_valid['text'].apply(preprocess)


# Vytvorenie vyváženého trénovacieho datasetu
df_one = my_train[my_train['label'] == 1]  
df_zero = my_train[my_train['label'] == 0]
# Náhodné zníženie počtu vzoriek triedy 1 na úroveň triedy 0
df_downsampled = resample(df_one, 
                                   replace=False,    
                                   n_samples=len(df_zero),  
                                   random_state=42)
df_balanced = pd.concat([df_downsampled, df_zero]) # Spojenie
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True) # Náhodné premiešanie
print("Po undersamplingu:")
print(df_balanced['label'].value_counts())


# Definovanie parametrov
# Počet unikátnych slov
all_words = []
for text in my_train['clean_text']:
    all_words.extend(text)  
word_counts = Counter(all_words)
maximum_features = len(word_counts) 
# Maximálna dĺžka sekvencie
max_length = max([len(text) for text in my_train['clean_text']])
# Ďalšie hyperparametre
embedding_dims = 50
no_of_filters = 100  
hidden_dims = 128
batch_size = 32  
epochs = 2 
threshold = 0.5 

# Tokenizácia a rozdelenie datasetu
tokenizer = Tokenizer(num_words=maximum_features)    
tokenizer.fit_on_texts(my_train['clean_text'])

x_train = tokenizer.texts_to_sequences(my_train['clean_text'])
x_train = pad_sequences(x_train, maxlen=max_length)
y_train = my_train['label']

x_test = tokenizer.texts_to_sequences(my_test['clean_text'])
x_test = pad_sequences(x_test, maxlen=max_length)
y_test = my_test['label']

x_valid = tokenizer.texts_to_sequences(my_valid['clean_text'])
x_valid = pad_sequences(x_valid, maxlen=max_length)
y_valid = my_valid['label']

# Nastavenie náhodného seed
set_seed(45)


# //////////////////// Model CNN /////////////////////

model_cnn = Sequential()
model_cnn.add(Embedding(maximum_features, embedding_dims, input_length=max_length))
model_cnn.add(Conv1D(no_of_filters, kernel_size = 3, padding='valid', activation='relu', strides=1))
model_cnn.add(BatchNormalization())
model_cnn.add(Flatten())    
model_cnn.add(Dense(hidden_dims, activation='relu'))
model_cnn.add(Dropout(0.3))
model_cnn.add(Dense(1, activation='sigmoid'))
model_cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_cnn.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid))

# Predikcia pravdepodobností
y_prob_cnn = model_cnn.predict(x_test)
y_pred_cnn = (y_prob_cnn > threshold).astype(int)
# Vyhodnotenie výkonnosti modelu
accuracy = accuracy_score(y_test, y_pred_cnn)
precision = precision_score(y_test, y_pred_cnn)
recall = recall_score(y_test, y_pred_cnn)
f1 = f1_score(y_test, y_pred_cnn)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

# Detailná správa o výkonnosti pre každú triedu 
print(classification_report(y_test, y_pred_cnn, target_names=["Negative", "Positive"]))

# Сonfusion matrix pre CNN
cm = confusion_matrix(y_test, y_pred_cnn)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Non-Toxic', 'Toxic'], yticklabels=['Non-Toxic', 'Toxic'])
plt.xlabel('Predikovaná')
plt.ylabel('Skutočná')
plt.title('Confusion Matrix')
plt.show()

# ROC krivka pre CNN
fpr, tpr, thresholds = roc_curve(y_test, y_prob_cnn)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Vybraný tweet
item_index = 74
item = x_test[item_index:item_index+1]  
true_label = y_test[item_index]

# Predikcia pre konkrétny tweet
pred_prob = model_cnn.predict(item)[0][0]  
pred_class = int(pred_prob > threshold)    
print(f"Model predpovedal triedu: {pred_class} (pravdepodobnosť: {pred_prob:.4f})")
print(f"Skutočná trieda: {true_label}")


# //////////////////// LRP pre CNN /////////////////////

input_layer = model_cnn.input
# Získanie výstupu z predposlednej vrstvy (pred sigmoidou)
output_wo_sigmoid = model_cnn.layers[-2].output
# Vytvorenie modelu bez výstupnej sigmoid vrstvy
model_wo_sigmoid = Model(inputs=input_layer, outputs=output_wo_sigmoid)
# Inicializácia LRP analyzátora (metóda alpha_2_beta_1)
analyzer = innvestigate.create_analyzer("lrp.alpha_2_beta_1", model_wo_sigmoid)
# Výber testovacieho vstupu
input_example = x_test[item_index:item_index+1]  
# Výpočet relevancií
relevance = analyzer.analyze(input_example) 
print("Shape of relevance:", relevance.shape)
print("Shape of relevance[0]:", relevance[0].shape)
print("Relevance[0] sample:", relevance[0][:10])

word_relevance = relevance[0]
print("min relevance:", np.min(word_relevance))
print("max relevance:", np.max(word_relevance))
# Normalizácia hodnôt relevance do rozsahu 0–1
word_relevance = (word_relevance - np.min(word_relevance)) / (np.max(word_relevance) - np.min(word_relevance) + 1e-10)
words = tokenizer.sequences_to_texts([x_test[item_index]])[0].split()
# Výpis slov a ich relevance skóre
for word, score in zip(words[:len(word_relevance)], word_relevance):
    print(f"{word:>15}: {score:.4f}")

def color_word(word, score):
    color = f"rgba(0, 0, 255, {min(abs(score), 1)})"  
    return f"<span style='background-color: {color}; padding:2px; margin:1px; border-radius:3px'>{word}</span>"

html_text = " ".join([color_word(w, s) for w, s in zip(words[:len(word_relevance)], word_relevance)])

with open("lrp_cnn_output.html", "w", encoding="utf-8") as f:
    f.write(html_text)

print("Súbor bol uložený ako lrp_cnn_output.html")





# //////////////////// Model LSTM /////////////////////

model_lstm = Sequential()
model_lstm.add(Embedding(input_dim=maximum_features, output_dim=embedding_dims, input_length=max_length))
model_lstm.add(LSTM(units=64, return_sequences=False)) 
model_lstm.add(Dropout(0.4))  
model_lstm.add(Dense(1, activation='sigmoid'))  
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=2, batch_size=batch_size)

# Predikcia pravdepodobnosti
y_prob_lstm = model_lstm.predict(x_test)
y_pred_lstm = (y_prob_lstm > threshold).astype(int)
# Vyhodnotenie výkonnosti modelu
accuracy = accuracy_score(y_test, y_pred_lstm)
precision = precision_score(y_test, y_pred_lstm)
recall = recall_score(y_test, y_pred_lstm)
f1 = f1_score(y_test, y_pred_lstm)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_lstm)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Non-Toxic', 'Toxic'], yticklabels=['Non-Toxic', 'Toxic'])
plt.xlabel('Predikovaná')
plt.ylabel('Skutočná')
plt.title('Confusion Matrix')
plt.show()

# ROC krivka
fpr, tpr, thresholds = roc_curve(y_test, y_prob_lstm)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Vybraný tweet
sample_index = 26  
first_input = x_test[sample_index:sample_index+1]  
true_label = y_test[sample_index]

# Predikcia pre konkrétny tweet
pred_prob = model_lstm.predict(first_input)[0][0]
pred_class = int(pred_prob > threshold)
print(f"Model predpovedal triedu: {pred_class} (pravdepodobnosť: {pred_prob:.4f})")
print(f"Skutočná trieda: {true_label}")


# //////////////////// SHAP pre LSTM /////////////////////

# Funkcia na dekódovanie sekvencie späť na slová pomocou tokenizeru
def decode_tokens(sequence, tokenizer):
    index_word = {v: k for k, v in tokenizer.word_index.items()}
    return [index_word.get(i, '[PAD]' if i == 0 else '[UNK]') if i not in index_word else index_word[i] for i in sequence]
# Získanie čitateľných tokenov
first_tokens = decode_tokens(x_test[sample_index], tokenizer)
# Výber náhodných vzoriek z trénovacej množiny pre KernelExplainer
explainer_samples = x_train[np.random.choice(len(x_train), size=100, replace=False)]
# Vytvorenie SHAP explainer
explainer = shap.KernelExplainer(model_lstm.predict, explainer_samples)

# Výpočet SHAP hodnôt
shap_values = explainer.shap_values(first_input)
print("SHAP shape:", np.array(shap_values).shape)
print("SHAP[0] shape:", np.array(shap_values)[0].shape)
print("SHAP[0][0] shape:", np.array(shap_values)[0][0].shape)


shap_vals = shap_values[0].squeeze(-1)
nonpad_idx = x_test[sample_index] != 0
first_tokens = decode_tokens(x_test[sample_index], tokenizer)
filtered_tokens = [t for i, t in enumerate(first_tokens) if nonpad_idx[i]]
filtered_vals = shap_vals[nonpad_idx]

# Kontrola počtu tokenov a SHAP hodnôt
print("Токенів:", len(filtered_tokens))
print("SHAP значень:", len(filtered_vals))

# Vytvorenie grafu
colors = ['blue' if v > 0 else 'red' for v in filtered_vals]
plt.figure(figsize=(12, 2))
plt.barh(range(len(filtered_tokens)), filtered_vals, color=colors, align='center')
plt.yticks(range(len(filtered_tokens)), filtered_tokens)
plt.xlabel("SHAP hodnoty")
plt.title("SHAP pre model LSTM")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()











