import random
import math
import argparse
from collections import Counter
from numpy.lib.stride_tricks import sliding_window_view

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

def score_template(template: str, k) -> tuple:
    return calculate_mean_p(template), expanded_motif_penalty(template, k)

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

def optimize_barcode_template(barcode_len, ks, num_designs=100, iterations=10000,
                              max_homopolymer_len=3, no_gquad=False, initial_temp=1.0, cooling_rate=0.0001, **kwargs):
    print(f'Generating {num_designs} optimized barcodes ')
    best_candidates = []
    for num in range(num_designs):
        current = build_initial_template(barcode_len, max_homopolymer_len, no_gquad)
        current_score = score_template(current, ks)
        candidates = [(current, current_score)]
        temp = initial_temp

        for _ in range(iterations):
            proposal = mutate(current, max_homopolymer_len, iterations, no_gquad)
            prop_score = score_template(proposal, ks)
            delta = prop_score[0]- current_score[0]
            # delta = prop_score[0]*prop_score[1] - current_score[0]*current_score[1]
            # delta = (prop_score[0]/(1+prop_score[1]) - current_score[0]/(1+current_score[1]))
            if delta >= 0 or random.random() < math.exp(delta / temp):
                current, current_score = proposal, prop_score
                candidates.append((current, current_score))
            temp = temp / (1 + cooling_rate * temp)

        best_candidates.append((current, current_score))

    return best_candidates

if __name__ == '__main__':
    # Optimize template barcodes to have diverse motifs (ks) and large combinatorial space
    parser = argparse.ArgumentParser()

    # Parameters if generating new barcode
    parser.add_argument('--barcode_len', type=int, default=60,
                        help='Length of barcode when generating')
    parser.add_argument('--max_homopolymer_len', type=int, default=4,
                        help='Do not allow sequences with possible homopolymers longer than this value')
    parser.add_argument('--iterations', type=int, default=25,
                        help='Simulated annealing iterations for each barcode template')
    parser.add_argument('--ks', type=float, default=[1,2,3], nargs='+',
                        help='size of windows to look over to assess sequence diversity/repetitiveness')
    parser.add_argument('--num_designs', type=float, default=1000,
                        help='How many times to try optimizing different barcode templates')
    parser.add_argument('--no_gquad', default=True, action='store_true',
                        help='Eliminate the possiblility of G quadraplexes by not allowing 3 consecutive gs'
                             'This is mostly important for RNA barcodes')

    args = parser.parse_args()

    best_candidates = optimize_barcode_template(args.barcode_len, args.ks, args.num_designs, args.iterations, args.max_homopolymer_len, args.no_gquad)
    best_candidates = [(x[0], adjust_p(x[1][0])) for x in best_candidates]
    best_candidates = sorted(best_candidates, key=lambda x: x[1])
    filtered_candidates = [i for i in best_candidates if all([n not in i[0] for n in nucleotides])]

    print('Full results: \n', best_candidates)
    print(f'Best candidate: {best_candidates[0][1], best_candidates[0][1]}\n')

    print('Full results filtered: \n', filtered_candidates)
    print(f'Best filtered candidate: {filtered_candidates[0][0], filtered_candidates[0][1]}')
