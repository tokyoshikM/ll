import json, subprocess
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import csv
r=Path(__file__).resolve().parent.parent
rows=list(csv.DictReader((r/'training'/'dataset.csv').open(encoding='utf-8')))
texts=[v['text'] for v in rows]
y=[int(v['label']=='synthetic') for v in rows]
v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),lowercase=True,max_features=3500,sublinear_tf=False,norm='l2')
x=v.fit_transform(texts)
m=LogisticRegression(max_iter=1000,random_state=42,class_weight='balanced').fit(x,y)
inputs=[texts[0],texts[-1],'Осенний ветер прогулялся у реки и старые деревья тихо разбудил под утро.', 'Далеко в полях зазеленела трава и речка обняла ночные огни.']
expected=m.predict_proba(v.transform(inputs))[:,1].tolist()
script="import {predict} from './model/predict.js'; import {readFileSync} from 'node:fs';const m=JSON.parse(readFileSync('./model/model.json','utf8'));for(const s of JSON.parse(process.argv[1])) console.log(predict(m,s).score);"
p=subprocess.run(['node','--input-type=module','-e',script,json.dumps(inputs,ensure_ascii=False)],cwd=r,capture_output=True,text=True,check=True)
actual=[float(z) for z in p.stdout.strip().splitlines()]
for a,b in zip(actual,expected):assert abs(a-b)<1e-9,(a,b)
print('PASS Python/JavaScript model equivalence,',len(actual),'examples')
