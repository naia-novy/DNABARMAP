import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations, product

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


def cluster_barcodes_hamming(barcodes, max_distance=1):
    """Cluster barcodes by Hamming distance ≤ max_distance using Union-Find."""
    barcode_list = list(set(barcodes))
    barcode_to_idx = {bc: i for i, bc in enumerate(barcode_list)}

    parent = list(range(len(barcode_list)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    existing = set(barcode_list)
    alphabet = 'ACGT'

    for bc in barcode_list:
        bc_idx = barcode_to_idx[bc]
        bc_list = list(bc)
        k = len(bc)

        for dist in range(1, max_distance + 1):
            for positions in combinations(range(k), dist):
                orig_nucs = [bc_list[p] for p in positions]

                for substitutions in product(alphabet, repeat=dist):
                    if any(substitutions[i] == orig_nucs[i] for i in range(dist)):
                        continue

                    for i, pos in enumerate(positions):
                        bc_list[pos] = substitutions[i]

                    neighbor = ''.join(bc_list)
                    if neighbor in existing:
                        union(bc_idx, barcode_to_idx[neighbor])

                    for i, pos in enumerate(positions):
                        bc_list[pos] = orig_nucs[i]

    barcode_to_cluster = {bc: find(barcode_to_idx[bc]) for bc in barcode_list}
    return barcode_to_cluster


def collapse_clusters(df, barcode_to_cluster):
    """Collapse clusters, keeping highest-count barcode and summing counts."""
    df = df.copy()
    df['cluster'] = df['barcode'].map(barcode_to_cluster)

    collapsed = []
    for cluster_id, group in df.groupby('cluster'):
        best_row = group.loc[group['counts'].idxmax()]
        coding_counts = group.groupby('coding_region')['counts'].sum()
        best_coding = coding_counts.idxmax()

        collapsed.append({
            'barcode': best_row['barcode'],
            'coding_region': best_coding,
            'counts': group['counts'].sum()
        })

    return pd.DataFrame(collapsed)


# =============================================================================
# Configuration
# =============================================================================

template_barcode = 'MNBRWHBWRYBYRYWNVYDRHKHSNDHKMRDWKDMBKWNVSWKWVNBVWKDVWDKVHVKNDHVKDMVHKHSKWBN'
barcode_len = len(template_barcode)

promoter_path = 'mapping/updated_control_promoters2.csv'

fns = ['mapping/barcode22_mapping.tsv',
       'mapping/barcode23_mapping.tsv']

fns = ['mappinga/barcode22_mapping.tsv',
       'mappinga/barcode23_mapping.tsv']
#
# fns = ['mapping_direct_illumina/mapping_22_direct_combined.tsv',
#        'mapping_direct_illumina/mapping_23_direct_combined.tsv']

fns = ['mapping_illumina/barcode22_mapping.tsv',
       'mapping_illumina/barcode23_mapping.tsv',]

fns = ['mapping_direct_illumina/barcode22_mapping.tsv',
       'mapping_direct_illumina/barcode23_mapping.tsv',]

count_cutoff = 5
hamming_distance = 1

RBS = 'TTTAACTTTAAGAAGGAGATATACAT'
promoter_options = pd.read_csv(promoter_path)
promoter_options.promoter = promoter_options.promoter + RBS

# =============================================================================
# Processing
# =============================================================================

for fn in fns:
    df = pd.read_csv(fn, sep='\t')
    print(len(df))
    df = df.drop(columns=['filename'])
    df.dropna(inplace=True)
    df = df.loc[df.barcode.str.len() == barcode_len]

    # Group by barcode + coding_region
    df_combined = df.groupby(['barcode', 'coding_region']).size().reset_index(name='counts')
    print(f"{fn}: {len(df_combined)} unique barcode+coding pairs")

    # Filter by frequency
    if df_combined.counts.mean() > 1.25:
        df_combined = df_combined.loc[df_combined.counts >= count_cutoff]
        print(f"After count filter: {len(df_combined)}")

        barcode_to_cluster = cluster_barcodes_hamming(
            df_combined['barcode'].tolist(),
            max_distance=hamming_distance
        )
        df_combined = collapse_clusters(df_combined, barcode_to_cluster)
        print(f"After Hamming-{hamming_distance} clustering: {len(df_combined)}")

    # Ensure barcodes match the template
    test = df_combined.loc[df_combined.barcode.apply(lambda x: checker(template_barcode, x))]

    if len(test) <= len(df_combined) * 0.05:
        print(f"Trying reverse complement for {fn}")
        # FIX: RC the barcodes only. Coding regions were already standardized
        # to forward orientation by map_barcodes(), so RC-ing them here would
        # double-invert them and cause barcode→coding mismatches across files.
        df_combined['barcode'] = df_combined['barcode'].apply(reverse_complement)
        df_combined = df_combined.loc[df_combined.barcode.apply(lambda x: checker(template_barcode, x))]
        # DO NOT RC coding_region here
    else:
        df_combined = test

    print(f"{fn}: {len(df_combined)} final clusters\n")
    sns.histplot(df_combined.counts)
    plt.title(fn)
    plt.show()

    df_combined.to_csv(fn.replace('.tsv', '_final.tsv'), sep='\t', index=False)