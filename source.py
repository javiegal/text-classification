import pandas as pd
from flair.data import Corpus
from flair.data import Sentence
from flair.datasets import ClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

# Create a DataFrame with clickbait and non-clickbait headlines
with open("data/clickbait_data") as file:
    cb_lines = [Sentence(line.rstrip()) for line in file if
                line.rstrip() != '']

with open("data/non_clickbait_data") as file:
    non_cb_lines = [Sentence(line.rstrip()) for line in file if
                    line.rstrip() != '']

cb = pd.DataFrame(zip(['__label__cb' for _ in range(len(cb_lines))], cb_lines),
                  columns=['text', 'label'])
non_cb = pd.DataFrame(
    zip(['__label__not cb' for _ in range(len(non_cb_lines))], non_cb_lines),
    columns=['text', 'label'])
df = pd.concat([non_cb, cb])
df = df.sample(frac=1).drop_duplicates()

df.iloc[:int(len(df) * 0.05)].to_csv('data/train.csv',
                                     sep='\t', index=False,
                                     header=False)
df.iloc[int(len(df) * 0.05):int(len(df) * 0.95)].to_csv('data/test.csv',
                                                        sep='\t', index=False,
                                                        header=False)
df.iloc[int(len(df) * 0.95):].to_csv('data/dev.csv',
                                     sep='\t', index=False,
                                     header=False)

corpus: Corpus = ClassificationCorpus(
    './data',
    test_file='test.csv',
    dev_file='dev.csv',
    train_file='train.csv')

document_embeddings = TransformerDocumentEmbeddings(fine_tune=True)

classifier = TextClassifier(document_embeddings,
                            label_dictionary=corpus.make_label_dictionary(),
                            multi_label=False)

trainer = ModelTrainer(classifier, corpus)
trainer.train('.', max_epochs=3, num_workers=16, learning_rate=0.001,
              embeddings_storage_mode='cpu')

classifier = TextClassifier.load('results/best-model.pt')
sentences = [Sentence(
    "16 Men Died in New York City Jails Last Year. Who Were They?"),
    Sentence(
        'Massive Price Change on this one!!! Almost NEW! Check it out – $259,900!!!'),
    Sentence('I CAN’T BELIEVE THIS HAPPENED… OMG!'),
    Sentence(
        'Federal Judge Rejects Hate-Crime Plea Deals in Ahmaud Arbery Killing'),
    Sentence("We Know Why You're Single Based On Your Zodiac Sign"),
    Sentence(
        'Covid-19: CDC warns against travel to 22 countries including Australia and Israel'),
    Sentence('Summer breaks: 20 of the best self-catering stays in the UK'),
    Sentence('Our choices for self-catering stays in the UK'),
    Sentence("Corona Centennial leads The Times' top 25 prep basketball rankings"),
    Sentence('18 Times When Stop Clickbait Has Saved the Day')
]

classifier.predict(sentences)
for sentence in sentences:
    print(sentence)
