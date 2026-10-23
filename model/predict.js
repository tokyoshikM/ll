/** Browser equivalent of sklearn TfidfVectorizer(analyzer='char',ngram_range=(3,5)) + LogisticRegression.
 * This demo classifier separates public-domain examples from template-generated examples,
 * NOT verified human versus LLM text. Do not present as a validated authorship detector.
 */
export function predict(model,text){
  if(!model || model.format!=='authorlab-char-tfidf-logreg-v1') throw new Error('Неизвестный формат модели');
  const s=text.toLowerCase();const counts=new Map();let total=0,matched=0;
  for(let n=model.ngramMin;n<=model.ngramMax;n++)for(let i=0;i<=s.length-n;i++){
    const gram=s.slice(i,i+n);total++;
    const j=model.vocabulary[gram];if(j===undefined)continue;
    matched++;counts.set(j,(counts.get(j)||0)+1);
  }
  let sumSq=0;const values=[];
  for(const [j,count] of counts){const val=count*model.idf[j];values.push([j,val]);sumSq+=val*val}
  const norm=Math.sqrt(sumSq);let logit=model.intercept;
  if(norm>0)for(const [j,val] of values)logit+=val/norm*model.weights[j];
  const score=1/(1+Math.exp(-Math.max(-50,Math.min(50,logit))));
  return {score,coverage:total?matched/total:0,features:counts.size,chars:s.length};
}
