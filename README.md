Téma práce: 
**Vysvetliteľnosť a použiteľnosť modelov dátovej analytiky na detekciu toxicity**

Cieľom tejto práce bolo klasifikovať text (odhaliť toxické komentáre) pomocou rôznych modelov hlbokého učenia, vrátane LSTM, CNN a BERT. Okrem toho práca obsahuje aj analýzu a porovnanie vysvetliteľnosti modelov pomocou nástrojov ako SHAP a LRP.


Použitý dataset:
Na trénovanie a testovanie modelov bol použitý verejne dostupný dataset **Tweet_Sentiment_pos_neg** z platformy **Hugging Face**. Dataset obsahuje texty tweetov označené ako pozitívne alebo negatívne, čo umožňuje binárnu klasifikáciu sentimentu.

Základné informácie:
- Zdroj: Hugging Face Datasets
- Tréningová množina: 12 900 vzoriek
- Testovacia množina: 3 700 vzoriek
- Validačná množina: 1 850 vzoriek
- Formát: CSV
- Stĺpce:
    - text – text tweetu
    - label – sentiment (0 = negatívny, 1 = pozitívny)


Štruktúra súborov:
- `data/`: obsahuje CSV súbory s údajmi (myds_train.csv, myds_test.csv, myds_valid.csv)
- `CNN_LSTM_LRP_SHAP.py`: obsahuje implementáciu modelov (CNN, LSTM) a metód vysvetliteľnosti (LRP, SHAP)
- `BERT_LRP.py`: obsahuje implementáciu modelu BERT a metódy vysvetliteľnosti LRP
- `README.md` 

Použité technológie a knižnice:
Projekt bol implementovaný v jazyku **Python 3.10.11** Na trénovanie modelov, predspracovanie textových dát, vyhodnocovanie výsledkov a vizualizáciu boli využité nasledovné knižnice:

Knižnica	        Verzia	        Popis
--------------------------------------------------------------------------
pandas	          2.2.3	          Práca s datasetmi a manipulácia dát
numpy	            1.24.3	        Numerické operácie
matplotlib	      3.10.1	        Vizualizácia grafov
tensorflow	      2.14.1	        Tréning modelov CNN a LSTM
torch       	    2.6.0	          Tréning BERT modelu
transformers	    4.50.1	        Načítanie predtrénovaného BERT modelu
scikit-learn	    1.6.1	          Vyhodnocovanie metrík a rozdelenie dát
shap	            0.47.1	        SHAP hodnoty
seaborn	          0.13.2	        Vizualizácia matice zámen
wordcloud	        1.9.4	          Generovanie wordcloud grafu
nltk	            3.9.1           Natural Language Toolkit – tokenizácia, stop slová, lematizácia
innvestigate	    2.1.0           Implementácia LRP 
