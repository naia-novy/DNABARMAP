import random
import math
from itertools import product
from collections import defaultdict
import argparse

import seaborn as sns
import matplotlib.pyplot as plt

# IUPAC definitions
IUPAC_CODES = {
    'A': ['A'],  'C': ['C'],  'G': ['G'],  'T': ['T'],
    'R': ['A','G'], 'Y': ['C','T'], 'S': ['G','C'], 'W': ['A','T'],
    'K': ['G','T'], 'M': ['A','C'], 'B': ['C','G','T'], 'D': ['A','G','T'],
    'H': ['A','C','T'], 'V': ['A','C','G'], 'N': ['A','C','G','T']
}
IUPAC_KEYS = list(IUPAC_CODES.keys())
DEGENERATE_KEYS = [k for k in IUPAC_KEYS if len(IUPAC_CODES[k]) > 1]
NUCLEOTIDES = ['A', 'C', 'G', 'T']

# Function to check potential homopolymers in a degenerate template
def could_form_homopolymer(template: str, max_homopolymer_len: int) -> bool:
    for nuc in NUCLEOTIDES:
        run = 0
        for c in template:
            if nuc in IUPAC_CODES.get(c, []):
                run += 1
                if run > max_homopolymer_len:
                    return True
            else:
                run = 0
    return False

# Objective functions
def sequence_entropy(template: str) -> float:
    return sum(math.log2(len(IUPAC_CODES[b])) for b in template)


def expanded_motif_penalty(template, ks):
    penalty = 0.0
    for k in ks:
        windows = []
        for i in range(len(template) - k + 1):
            pools = [IUPAC_CODES[c] for c in template[i:i+k]]
            poss = {''.join(p) for p in product(*pools)}
            windows.append(poss)
        motif_counts = defaultdict(int)
        for poss in windows:
            for motif in poss:
                motif_counts[motif] += 1
        penalty += sum((cnt - 1) for cnt in motif_counts.values() if cnt > 1)

    return penalty / len(ks)

def score_template(template: str, k) -> tuple:
    return sequence_entropy(template), expanded_motif_penalty(template, k)

def dominates(a: tuple, b: tuple) -> bool:
    return (a[0] >= b[0] and a[1] <= b[1]) and (a[0] > b[0] or a[1] < b[1])

def pareto_front(candidates: list) -> list:
    front = []
    for tpl, score in candidates:
        if not any(dominates(other_score, score) for _, other_score in candidates if other_score != score):
            front.append((tpl, score))
    return front

# Build template biased to high entropy but avoid single-nucleotide codes
def build_initial_template(length, max_homopolymer_len):
    template = ''
    for _ in range(length):
        # try degenerate codes first
        choices = [c for c in DEGENERATE_KEYS
                   if not could_form_homopolymer(template + c, max_homopolymer_len)]

        if not choices:
            template = template[:-1]
            continue
        weights = [len(IUPAC_CODES[c]) for c in choices]
        template += random.choices(choices, weights)[0]
    return template

# Mutate biased to increase entropy but prefer degenerate
def mutate(template, max_homopolymer_len, iterations):
    for _ in range(iterations):
        pos = random.randrange(len(template))
        # prefer degenerate replacements
        alt_deg = [c for c in DEGENERATE_KEYS if c != template[pos]]
        valid = [c for c in alt_deg
                 if not could_form_homopolymer(template[:pos] + c + template[pos+1:], max_homopolymer_len)]
        if not valid:
            # fallback to any code
            alt_all = [c for c in DEGENERATE_KEYS if c != template[pos]]
            valid = [c for c in alt_all
                     if not could_form_homopolymer(template[:pos] + c + template[pos+1:], max_homopolymer_len)]
        if not valid:
            continue
        weights = [len(IUPAC_CODES[c]) for c in valid]
        new_c = random.choices(valid, weights)[0]
        return template[:pos] + new_c + template[pos+1:]
    return template


def pick_elbow_candidate(pareto_candidates: list) -> tuple:
    # Prepare points: sort by penalty ascending
    pts = sorted(pareto_candidates, key=lambda x: x[1][1])
    # Convert to numeric arrays
    xs = [pt[1][1] for pt in pts]  # penalty
    ys = [pt[1][0] for pt in pts]  # entropy
    # Line endpoints
    x1, y1 = xs[0], ys[0]
    x2, y2 = xs[-1], ys[-1]
    # Compute distances
    max_dist = -1
    elbow_idx = 0
    for i, (x0, y0) in enumerate(zip(xs, ys)):
        # distance from (x0,y0) to line through (x1,y1)-(x2,y2)
        num = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
        den = math.hypot(y2 - y1, x2 - x1)
        dist = num/den if den else 0
        if dist > max_dist:
            max_dist = dist
            elbow_idx = i
    return pts[elbow_idx]


def optimize_barcode_template(length, ks, num_designs=100, iterations=10000,
                              max_homopolymer_len=3, initial_temp=1.0, cooling_rate=0.0001):
    print(f'Generating {num_designs} optimized barcodes ')
    best_candidates = []
    for num in range(num_designs):
        current = build_initial_template(length, max_homopolymer_len)
        current_score = score_template(current, ks)
        candidates = [(current, current_score)]
        temp = initial_temp

        for _ in range(iterations):
            proposal = mutate(current, max_homopolymer_len, iterations)
            prop_score = score_template(proposal, ks)
            delta = (prop_score[0]/(1+prop_score[1]) - current_score[0]/(1+current_score[1]))

            if delta >= 0 or random.random() < math.exp(delta / temp):
                current, current_score = proposal, prop_score
                candidates.append((current, current_score))
            temp = temp / (1 + cooling_rate * temp)

        best_candidates.append((current, current_score))

    return pareto_front(best_candidates)


if __name__ == '__main__':
    # Simple simulated annealing script to optimize barcode templates to balance tradeoff between diversity and entropy
    # Select barcode at elbow of this pareto front
    parser = argparse.ArgumentParser()

    # Parameters if generating new barcode
    parser.add_argument('--barcode_len', type=int, default=60,
                        help='Length of barcode when generating')
    parser.add_argument('--max_homopolymer_len', type=int, default=3,
                        help='Do not allow sequences with possible homopolymers longer than this value')
    parser.add_argument('--iterations', type=int, default=300,
                        help='Simulated annealing iterations for each barcode template')
    parser.add_argument('--ks', type=float, default=[2,3,4,5,6,7], nargs='+',
                        help='size of windows to look over to assess sequence diversity/repetitiveness')
    parser.add_argument('--num_designs', type=float, default=300,
                        help='How many times to try optimizing different barcode templates')

    args = parser.parse_args()

    pareto = optimize_barcode_template(args.barcode_len, args.ks, args.num_designs, args.iterations, args.max_homopolymer_len)
    x = [i[1][0] for i in pareto]
    y = [i[1][1] for i in pareto]
    sns.scatterplot(x=x,y=y)
    plt.show()

    best_candidate = pick_elbow_candidate(pareto)
    print('Full results: \n', pareto)
    print(best_candidate)

    print(f'\nBest candidate: {best_candidate[0]}')

