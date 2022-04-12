import json
with open('database.json', 'r', encoding='utf-8') as f:
    db = json.load(f)
knowledges = []
db_keys = []
for w, dit in db['nameDict'].items():
    for k in dit['word']:
        k1 = w + '是' + k + '的' + w
        k2 = k + '的' + w
        knowledges.append(k)
        knowledges.append(k1)
        knowledges.append(k2)
        db_keys.append(w)
        db_keys.append(w)
        db_keys.append(w)
    for k in dit['pack']:
        knowledges.append(k)
        db_keys.append(w)

with open('knowledge_db.json', 'w', encoding='utf-8') as f:
    json.dump(knowledges, f, indent=2, ensure_ascii=False)

with open('db_key.json', 'w', encoding='utf-8') as f:
    json.dump(db_keys, f, indent=2, ensure_ascii=False)