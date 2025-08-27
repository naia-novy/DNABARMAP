import random
import math
import argparse
from collections import Counter
from numpy.lib.stride_tricks import sliding_window_view

import seaborn as sns
import matplotlib.pyplot as plt

from dnabarmap.utils import nuc_dict, import_cupy_numpy
np = import_cupy_numpy()

nuc_keys = list(nuc_dict.keys())
degenerate_keys = [k for k in nuc_keys if len(nuc_dict[k]) > 1]
nucleotides = ['A', 'C', 'G', 'T']

# Function to check potential homopolymers in a degenerate template
def could_form_homopolymer(template, max_homopolymer_len):
    for nuc in nucleotides:
        run = 0
        for c in template:
            if nuc in nuc_dict.get(c, []):
                run += 1
                if run > max_homopolymer_len:
                    return True
            else:
                run = 0
    return False

def adjust_p(p):
    return p
    adj_p = (1/p)

    return adj_p

def calculate_mean_p(template, adjust=True):
    obs = []
    for i, v in enumerate(template):
        obs.append(len(nuc_dict[v]))
    obs = Counter(obs)
    p = 0
    for k, v in obs.items():
        p += v / k
    p /= len(template)

    if adjust:
        return adjust_p(p)
    else:
        return p

def score_template(template, k):
    return expanded_motif_penalty(template, k), calculate_mean_p(template)


def dominates(a: tuple, b: tuple) -> bool:
    return (a[0] >= b[0] and a[1] <= b[1]) and (a[0] > b[0] or a[1] < b[1])

def pareto_front(candidates: list) -> list:
    front = []
    for tpl, score in candidates:
        if not any(dominates(other_score, score) for _, other_score in candidates if other_score != score):
            front.append((tpl, score))
    return front

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

def expanded_motif_penalty(template, ks):
    _BASE = {'A': 1, 'C': 2, 'G': 4, 'T': 8}
    _POP = np.array([bin(i).count("1") for i in range(16)], dtype=np.uint8)
    _eps = 1e-12

    masks = np.fromiter((sum(_BASE[b] for b in nuc_dict[c]) for c in template), dtype=np.uint8)
    L = len(masks)
    pens = []
    for k in ks:
        if k > L:
            pens.append(0.0); continue
        Wm = sliding_window_view(masks, k)
        cnts = np.prod(_POP[Wm].astype(float), axis=1)
        tot = 0.0
        for i in range(len(cnts)-1):
            inter = Wm[i] & Wm[i+1:]
            numer = np.prod(_POP[inter].astype(float), axis=1)
            denom = cnts[i] * cnts[i+1:]
            valid = denom > 0
            if np.any(valid):
                tot += np.sum(numer[valid] / (denom[valid] + _eps))
        pens.append(tot)
    return float(np.log1p(np.mean(pens)))

def check_conditions(template, max_homopolymer_len, no_gquad):
    result = could_form_homopolymer(template, max_homopolymer_len)
    if no_gquad:
        result = result or 'GGG' in template

    return result

def build_initial_template(length, max_homopolymer_len, no_gquad):
    # Build template biased to high entropy but avoid single-nucleotide codes
    generating = True
    counter = 0
    while generating:
        template = ''
        for _ in range(length):
            # try degenerate codes first
            choices = [c for c in degenerate_keys
                       if not check_conditions(template + c, max_homopolymer_len, no_gquad)]

            if not choices:
                # allow single codes if no degenerate valid
                choices = [c for c in nuc_keys
                           if not check_conditions(template + c, max_homopolymer_len, no_gquad)]
            if not choices:
                break
            weights = [len(nuc_dict[c]) for c in choices]
            template += random.choices(choices, weights)[0]

        if len(template) == length:
            generating = False
        counter += 1
        if counter > 100:
            raise Exception('Could not generate templates, try increasing max_homopolymer_len')
    return template

def mutate(template, max_homopolymer_len, iterations, no_gquad):
    # Mutate biased to increase entropy but prefer degenerate
    for _ in range(iterations):
        pos = random.randrange(len(template))
        alt_deg = [c for c in degenerate_keys if c != template[pos]]
        valid = [c for c in alt_deg
                 if not check_conditions(template[:pos] + c + template[pos+1:], max_homopolymer_len, no_gquad)]

        if not valid:
            continue
        weights = [len(nuc_dict[c]) for c in valid]
        new_c = random.choices(valid, weights)[0]
        return template[:pos] + new_c + template[pos+1:]
    return template


def optimize_barcode_template(barcode_len, ks, initial_designs, opt_frac, iterations=1000,
                              max_homopolymer_len=3, no_gquad=False, initial_temp=0.1, **kwargs):
    num_designs = int(initial_designs * opt_frac)
    print(f'Generating {num_designs} optimized barcodes ')

    temps = np.linspace(start=initial_temp, stop=0.000001, num=iterations)
    designs = []
    for num in range(initial_designs):
        current = build_initial_template(barcode_len, max_homopolymer_len, no_gquad)
        current_score = score_template(current, ks)
        designs.append([current, current_score])

    filtered_candidates = sorted(designs, key=lambda x: x[1][0])[:num_designs]

    best_candidates = []
    for d in filtered_candidates:
        d = d[0]
        current_score = score_template(d, ks)
        candidates = [(d, current_score)]
        temp = initial_temp

        track = [current_score[0]]
        for i in range(iterations):
            proposal = mutate(d, max_homopolymer_len, iterations, no_gquad)
            prop_score = score_template(proposal, ks)
            # delta = prop_score[0]*prop_score[1] - current_score[0]*current_score[1]
            delta = (prop_score[0]/(1+prop_score[1]) - current_score[0]/(1+current_score[1]))

            if delta >= 0 or random.random() < math.exp(delta / temp):
                d, current_score = proposal, prop_score
                candidates.append((d, current_score))
            track.append(current_score[0])
            temp = temps[i]

        best_candidates.append((d, current_score))

        # sns.lineplot(track)
        # plt.show()

        # best = sorted(candidates, key=lambda x: x[1][0])[-1]
        # best_candidates.append(best)

    return best_candidates


def cli():
    # Optimize template barcodes to have diverse motifs (ks) and large combinatorial space
    parser = argparse.ArgumentParser()

    # Parameters if generating new barcode
    parser.add_argument('--barcode_len', type=int, default=60,
                        help='Length of barcode when generating')
    parser.add_argument('--max_homopolymer_len', type=int, default=4,
                        help='Do not allow sequences with possible homopolymers longer than this value')
    parser.add_argument('--iterations', type=int, default=5000,
                        help='Simulated annealing iterations for each barcode template')
    parser.add_argument('--ks', type=int, default=[1,2,3,4,5], nargs='+',
                        help='size of windows to look over to assess sequence diversity/repetitiveness')
    parser.add_argument('--initial_designs', type=int, default=200,
                        help='How many times to try optimizing different barcode templates')
    parser.add_argument('--opt_frac', type=float, default=0.75,
                        help='How many times to try optimizing different barcode templates')
    parser.add_argument('--no_gquad', default=True, action='store_true',
                        help='Eliminate the possiblility of G quadraplexes by not allowing 3 consecutive gs'
                             'This is mostly important for RNA barcodes')

    args = parser.parse_args()

    candidates = optimize_barcode_template(**vars(args))
    pareto_candidates = pareto_front(candidates)
    # best_candidates = sorted(candidates, key=lambda x: x[1][0]*x[1][1])
    # best_candidates = [(x[0], adjust_p(x[1][0])) for x in best_candidates]
    best_candidates = sorted(pareto_candidates, key=lambda x: x[1][0]*x[1][1])
    # best_candidates = [(x[0], x[1][0]) for x in best_candidates]
    filtered_candidates = [(i[0], (adjust_p(i[1][0]), i[1][1]))  for i in best_candidates if all([n not in i[0] for n in nucleotides])]

    # print('Full results: \n', best_candidates)
    # print(f'Best candidate: {best_candidates[0][1], best_candidates[0][1]}\n')

    print('Full filtered: \n', filtered_candidates)
    print(f'Best candidate: {filtered_candidates[0][0], filtered_candidates[0][1]}')

    x = [i[1][0] for i in filtered_candidates]
    y = [i[1][1] for i in filtered_candidates]
    sns.scatterplot(x=x, y=y)
    plt.show()

    elbow_candidate = pick_elbow_candidate(filtered_candidates)
    print(f'Elbow candidate: {elbow_candidate[0], elbow_candidate[1]}')

if __name__ == '__main__':
    cli()