import json

with open('backend/output/drug_sentiment_summary.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'Total drugs tracked: {len(data)}')

print('\nSample drug sentiment:')
for i, (drug, info) in enumerate(list(data.items())[:5]):
    print(f'{drug}: {info["positive_count"]} positive, {info["negative_count"]} negative')