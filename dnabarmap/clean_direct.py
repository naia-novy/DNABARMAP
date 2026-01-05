import pandas as pd
import subprocess
from Bio import SeqIO
import seaborn as sns
import matplotlib.pyplot as plt

from dnabarmap.utils import nuc_dict


def reverse_complement(seq):
    complement = {
        'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
        'R': 'Y', 'Y': 'R',
        'S': 'S', 'W': 'W',
        'K': 'M', 'M': 'K',
        'B': 'V', 'V': 'B',
        'D': 'H', 'H': 'D',
        'N': 'N'
    }
    return ''.join(complement[b] for b in reversed(seq.upper()))


def checker(barcode_template, test_barcode):
    result = [test_barcode[i] in nuc_dict[v] for i, v in enumerate(barcode_template)]
    return sum(result) == len(result)


def get_cluster_consensus(obs_string, records):
    """Get most frequent barcode and coding region from cluster members."""
    indices = [int(x) - 1 for x in obs_string.split(',')]  # starcode is 1-indexed
    barcodes = [str(records[i].seq) for i in indices]
    coding_regions = [records[i].id for i in indices]

    consensus_barcode = max(set(barcodes), key=barcodes.count)
    consensus_coding = max(set(coding_regions), key=coding_regions.count)
    return consensus_barcode, consensus_coding


template_barcode = 'MNBRWHBWRYBYRYWNVYDRHKHSNDHKMRDWKDMBKWNVSWKWVNBVWKDVWDKVHVKNDHVKDMVHKHSKWBN'
barcode_len = len(template_barcode)

promoter_path = 'updated_control_promoters2.csv'

# fns = ['mapping_direct_illumina/mapping_direct_small-promoter_joined.tsv',
#        'mapping_direct_illumina/mapping_direct_large-promoter_joined.tsv']
# count_cutoff = 25

fns = ['mapping/barcode22_mapping.tsv',
       'mapping/barcode23_mapping.tsv']
count_cutoff = 1


RBS = 'TTTAACTTTAAGAAGGAGATATACAT'
promoter_options = pd.read_csv(promoter_path)
promoter_options.promoter = promoter_options.promoter + RBS


for fn in fns:
    barcode_fn = fn.replace('.tsv', '.fasta')
    output_log = fn.replace('.tsv', '.txt')

    df = pd.read_csv(fn, sep='\t')
    df = df.drop(columns=['filename'])
    df.dropna(inplace=True)
    df = df.loc[df.barcode.str.len() == barcode_len]

    # Group
    df_combined = df.groupby(['barcode', 'coding_region']).size().reset_index(name='counts')
    print(len(df_combined))

    # Filter by frequency
    df_combined = df_combined.loc[df_combined.counts >= count_cutoff]
    print(len(df_combined))

    # Ensure barcodes match the template
    test = df_combined.loc[df_combined.barcode.apply(lambda x: checker(template_barcode, x))]

    if len(test) <= len(df_combined) * 0.05:
        print(f"Trying reverse complement for {fn}")
        df_combined['barcode'] = df_combined['barcode'].apply(reverse_complement)
        df_combined = df_combined.loc[df_combined.barcode.apply(lambda x: checker(template_barcode, x))]
        df_combined['coding_region'] = df_combined['coding_region'].apply(reverse_complement)
    else:
        df_combined = test

    print(f"{fn}: {len(df_combined)} clusters")
    sns.histplot(df_combined.counts)
    plt.title(fn)
    plt.show()

    df_combined.to_csv(fn.replace('.tsv', '_final.tsv'), sep='\t', index=False)


#
# for fn in fns:
#     barcode_fn = fn.replace('.tsv', '.fasta')
#     output_log = fn.replace('.tsv', '.txt')
#
#     df = pd.read_csv(fn, sep='\t')
#     df = df.drop(columns=['filename'])
#     df.dropna(inplace=True)
#     df = df.loc[df.barcode.str.len() == barcode_len]
#
#     # Write fasta for starcode
#     # lines = '>' + df['coding_region'].astype(str) + '\n' + df['barcode'].astype(str) + '\n'
#     # with open(barcode_fn, "w") as f:
#     #     f.write(''.join(lines))
#
#     cmd = [
#         "/usr/local/bin/starcode",
#         "-d", "2",
#         "-r", '2',
#         '--seq-id',
#         '-o', output_log,
#         '-i', barcode_fn
#     ]
#     subprocess.run(cmd, stderr=subprocess.PIPE, check=True)
#
#     # Load starcode output and original records
#     cdf = pd.read_csv(output_log, sep='\t', header=None, names=['barcode', 'counts', 'observations'])
#     records = list(SeqIO.parse(barcode_fn, 'fasta'))
#
#     # Compute consensus for each cluster
#     consensus_data = cdf.apply(
#         lambda row: pd.Series(get_cluster_consensus(row['observations'], records)),
#         axis=1
#     )
#     cdf['consensus_barcode'] = consensus_data[0]
#     cdf['consensus_coding'] = consensus_data[1]
#
#     # Use consensus values instead of starcode representatives
#     df_combined = cdf[['consensus_barcode', 'consensus_coding', 'counts']].copy()
#     df_combined.columns = ['barcode', 'coding_region', 'counts']
#
#     # Ensure barcodes match the template
#     test = df_combined.loc[df_combined.barcode.apply(lambda x: checker(template_barcode, x))]
#
#     if len(test) <= len(df_combined) * 0.05:
#         print(f"Trying reverse complement for {fn}")
#         df_combined['barcode'] = df_combined['barcode'].apply(reverse_complement)
#         df_combined = df_combined.loc[df_combined.barcode.apply(lambda x: checker(template_barcode, x))]
#         df_combined['coding_region'] = df_combined['coding_region'].apply(reverse_complement)
#     else:
#         df_combined = test
#
#     # Filter by frequency
#     df_combined = df_combined.loc[df_combined.counts >= count_cutoff]
#
#     print(f"{fn}: {len(df_combined)} clusters")
#     sns.histplot(df_combined.counts)
#     plt.title(fn)
#     plt.show()
#
#     df_combined.to_csv(fn.replace('.tsv', '_final.tsv'), sep='\t', index=False)