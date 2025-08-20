import pandas as pd

true = pd.read_csv('/Users/natenovy/Research/DNABARMAP/syndata/syndataD_mapping_synthetic.tsv', sep='\t')
gen = pd.read_csv('/Users/natenovy/Research/DNABARMAP/DNABARMAP_outputs/syndataD_mapping.tsv', sep='\t')


correct = 0
for i, r in gen.iterrows():
    if r.barcode in list(true.true_barcode):
        correct += 1

correct = 0
for i, r in gen.iterrows():
    if r.barcode in list(true.true_barcode):
        hit = true.loc[true.true_barcode == r.barcode].variant.values[0]
        if hit == r.coding_region:
            correct += 1

print(correct)
