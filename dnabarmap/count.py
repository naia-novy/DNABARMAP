import pandas as pd
from Bio import SeqIO
from collections import Counter
import subprocess


from dnabarmap.utils import nuc_dict

def checker(barcode_template, test_barcode):
    result = [test_barcode[i] in nuc_dict[v] for i, v in enumerate(barcode_template)]
    return sum(result) == len(result)


# Usage
barcode = '23'
barcode_fn = f'barcodes_{barcode}.fasta'
input_tsv = f'mapping_{barcode}_combined.tsv'
output_fasta = f'out_{barcode}.fasta'
output_log = f'out_{barcode}.txt'
count_cutoff = 5
template_barcode = 'MNBRWHBWRYBYRYWNVYDRHKHSNDHKMRDWKDMBKWNVSWKWVNBVWKDVWDKVHVKNDHVKDMVHKHSKWBN'
promoter_path = 'updated_control_promoters2.csv'

df = pd.read_csv(input_tsv, sep='\t')
df.dropna(inplace=True)
df = df.loc[df.barcode.str.len() == 75]

with open(barcode_fn, "w") as f:
    for i,r  in df.iterrows():
        f.write(f">{r.coding_region}\n{str(r.barcode)}\n")

cmd = [
    "starcode",
    "-d", "5",
    "-r", "3",
    '--seq-id',
    '-o', output_log,
    '-i', barcode_fn]

subprocess.run(cmd, stderr=subprocess.PIPE, check=True)


df = pd.read_csv(output_log, sep='\t')
df = pd.DataFrame(df.values, columns=['barcode', 'counts', 'observations'])
df = df.loc[df.counts >= count_cutoff]
records = list(SeqIO.parse(barcode_fn, barcode_fn.split('.')[-1]))


mapping = {}
for i,r in df.iterrows():
    observations = []
    for o in r.observations.split(','):
        observations.append(records[int(o)-1].id)

    result = Counter(observations)
    final = result.most_common()[0]
    if final[1] >= sum(result.values()) / 3:
        mapping[r.barcode] = final[0]


# require correct barcode architecture
mapping = {k:v for k,v in mapping.items() if checker(template_barcode,k)}


# require known promoter sequence
RBS = 'TTTAACTTTAAGAAGGAGATATACAT'
promoter_options = pd.read_csv(promoter_path)
promoter_options.promoter = promoter_options.promoter + RBS

mapping = {k:v for k,v in mapping.items() if v in promoter_options.promoter.values}
count_mapping = {k:df.loc[df.barcode==k].counts.values[0] for k in mapping.keys()}



final_data = []
for k,v in mapping.items():
    final_data.append([k,v,count_mapping[k]])


final_data = pd.DataFrame(final_data, columns=['barcode','coding_region','counts'])
final_data.to_csv(f'direct_mapping_{barcode}.csv', index=False)


