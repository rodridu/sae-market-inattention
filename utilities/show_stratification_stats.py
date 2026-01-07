import pandas as pd

df = pd.read_parquet('C:/Users/ofs4963/Dropbox/Arojects/SAE/data/sentences_sae_train.parquet')

print('='*60)
print('STRATIFIED SAMPLE SUMMARY')
print('='*60)
print(f'\nTotal sentences: {len(df):,}')
print(f'Unique firms (CIK): {df["cik"].nunique():,}')
print(f'Unique documents: {df["accession_number"].nunique():,}')

print('\n--- Firm-Level Distribution ---')
firm_counts = df[df["cik"].notna()].groupby('cik').size()
print(f'Sentences per firm:')
print(f'  Mean:   {firm_counts.mean():.1f}')
print(f'  Median: {firm_counts.median():.1f}')
print(f'  Std:    {firm_counts.std():.1f}')
print(f'  Min:    {firm_counts.min()}')
print(f'  Max:    {firm_counts.max()}')
print(f'  Q1:     {firm_counts.quantile(0.25):.0f}')
print(f'  Q3:     {firm_counts.quantile(0.75):.0f}')

print('\n--- Year Distribution ---')
year_dist = df.groupby('year').size()
print(year_dist.to_string())

print(f'\n--- Item Distribution ---')
item_dist = df.groupby('item_type').size()
print(item_dist.to_string())

print('\n--- Top 10 Firms by Sentence Count ---')
top_firms = firm_counts.nlargest(10)
for cik, count in top_firms.items():
    print(f'  CIK {cik}: {count} sentences')

print('\n--- Bottom 10 Firms by Sentence Count ---')
bottom_firms = firm_counts.nsmallest(10)
for cik, count in bottom_firms.items():
    print(f'  CIK {cik}: {count} sentences')
