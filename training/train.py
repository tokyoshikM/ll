"""Train demo text classifier and export exact scikit-learn TF-IDF + logistic weights for JS."""
import csv, json, math
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

root=Path(__file__).resolve().parent.parent
rows=list(csv.DictReader((root/'training'/'dataset.csv').open(encoding='utf-8')))
texts=[r['text'] for r in rows]; labels=[1 if r['label']=='synthetic' else 0 for r in rows]
if len(set(labels)) != 2: raise ValueError('Need both human and synthetic classes')
vec=TfidfVectorizer(analyzer='char',ngram_range=(3,5),lowercase=True,max_features=3500,sublinear_tf=False,norm='l2')
X=vec.fit_transform(texts)
clf=LogisticRegression(max_iter=1000,random_state=42,class_weight='balanced').fit(X,labels)
terms=vec.get_feature_names_out()
artifact={'format':'authorlab-char-tfidf-logreg-v1','version':'demo-1','classes':['human','synthetic'],
 'labelDisclosure':'Synthetic samples are templated, not verified LLM generations; NOT an AI detection benchmark.',
 'ngramMin':3,'ngramMax':5,'vocabulary':{t:i for i,t in enumerate(terms)},
 'idf':vec.idf_.round(12).tolist(),'weights':clf.coef_[0].round(12).tolist(),
 'intercept':round(float(clf.intercept_[0]),12),
 'trainingRows':len(rows),'humanRows':sum(1 for y in labels if not y),'syntheticRows':sum(labels)}
(root/'model'/'model.json').write_text(json.dumps(artifact,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
print(f'Exported {len(terms)} features, {len(rows)} examples')
