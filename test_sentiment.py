import sys
sys.path.append('backend')
from modules.pipeline import run_pipeline
from modules.data_loader import load_webmd_reviews, load_indian_substitutes

reviews_df = load_webmd_reviews('backend/data/DS1_WebMD_Reviews.xlsx')
substitutes_dict = load_indian_substitutes('backend/data/DS2_IndianSubstitutes.xlsx')
medicine_reference = {'zosert': {'anxiety', 'anger', 'depression'}}

adr_results, drug_sentiment = run_pipeline(reviews_df, substitutes_dict, medicine_reference, limit=10)

print(f'ADR entries: {len(adr_results)}')
print(f'Drugs tracked: {len(drug_sentiment)}')
print(f'Zosert sentiment: {drug_sentiment.get("zosert", "Not found")}')