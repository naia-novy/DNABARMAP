import random
import math
import argparse
from functools import lru_cache
from numpy.lib.stride_tricks import sliding_window_view

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from dnabarmap.utils import nuc_dict, import_cupy_numpy
np = import_cupy_numpy()

nuc_keys = list(nuc_dict.keys())
degenerate_keys = [k for k in nuc_keys if len(nuc_dict[k]) > 1]
nucleotides = ['A', 'C', 'G', 'T']
_BASE = {'A': 1, 'C': 2, 'G': 4, 'T': 8}
_POP = np.array([bin(i).count("1") for i in range(16)], dtype=np.uint8)
_COMPLEMENT_MASK = np.array([
    ((mask & 1) << 3) |
    ((mask & 2) << 1) |
    ((mask & 4) >> 1) |
    ((mask & 8) >> 3)
    for mask in range(16)
], dtype=np.uint8)

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

def mean_degeneracy(template):
    return float(np.mean([len(nuc_dict[v]) for v in template]))


def _template_masks(template):
    return np.fromiter(
        (sum(_BASE[b] for b in nuc_dict[c]) for c in template),
        dtype=np.uint8,
    )


def _mean_mask_overlap(left_masks, right_masks):
    if len(left_masks) == 0 or len(right_masks) == 0:
        return 0.0
    overlap = _POP[left_masks & right_masks].astype(float)
    denom = np.minimum(_POP[left_masks], _POP[right_masks]).astype(float)
    valid = denom > 0
    if not np.any(valid):
        return 0.0
    return float(np.mean(overlap[valid] / denom[valid]))


@lru_cache(maxsize=32768)
def _template_metrics(template, max_shift=10):
    spans = np.fromiter((len(nuc_dict[c]) for c in template), dtype=float)
    masks = _template_masks(template)

    shift_best = 0.0
    rc_best = 0.0
    if len(masks) >= 2:
        rc_masks = _COMPLEMENT_MASK[masks[::-1]]
        limit = min(max_shift, len(masks) - 1)
        for shift in range(1, limit + 1):
            shift_best = max(
                shift_best,
                _mean_mask_overlap(masks[:-shift], masks[shift:]),
            )
            rc_best = max(
                rc_best,
                _mean_mask_overlap(masks[:-shift], rc_masks[shift:]),
            )

    return {
        'three_way_fraction': float(np.mean(spans == 3)),
        'four_way_fraction': float(np.mean(spans == 4)),
        'high_degeneracy_penalty': float(0.12 * np.mean(spans == 3) + 0.3 * np.mean(spans == 4)),
        'shift_similarity': shift_best,
        'rc_similarity': rc_best,
    }


def shift_similarity_penalty(template, max_shift=10):
    return _template_metrics(template, max_shift=max_shift)['shift_similarity']


def reverse_complement_similarity_penalty(template, max_shift=10):
    return _template_metrics(template, max_shift=max_shift)['rc_similarity']


def recoverability_penalty(template):
    metrics = _template_metrics(template)
    return (
        metrics['high_degeneracy_penalty']
        + 0.2 * metrics['shift_similarity']
        + 0.15 * metrics['rc_similarity']
    )


def seed_collision_penalty(template, ks=(6, 8, 10)):
    masks = _template_masks(template)
    spans = _POP[masks].astype(float)
    length = len(masks)
    penalties = []
    eps = 1e-12

    for k in ks:
        if k > length:
            continue

        window_masks = sliding_window_view(masks, k)
        window_spans = sliding_window_view(spans, k)

        same_position = np.prod(1.0 / np.maximum(window_spans, 1.0), axis=1)
        total = float(np.sum(same_position))

        for idx in range(len(window_masks) - 1):
            inter = _POP[window_masks[idx] & window_masks[idx + 1:]].astype(float)
            denom = window_spans[idx] * window_spans[idx + 1:]
            pair = np.prod(inter / (denom + eps), axis=1)
            total += float(np.sum(pair))

        norm = len(window_masks) + (len(window_masks) * (len(window_masks) - 1) / 2)
        penalties.append(total / max(norm, 1.0))

    if not penalties:
        return 0.0

    return float(np.log1p(np.mean(penalties) * 1e4))


def ambiguity_penalty(template, ks_tuple):
    return (
        0.35 * expanded_motif_penalty(template, ks_tuple)
        + recoverability_penalty(template)
    )


@lru_cache(maxsize=32768)
def _score_template_cached(template, ks_tuple):
    return (
        mean_degeneracy(template),
        ambiguity_penalty(template, ks_tuple),
    )


def score_template(template, ks):
    return _score_template_cached(template, tuple(ks))


def recoverability_excess_penalty(template,
                                  max_three_way_fraction=0.4,
                                  max_four_way_fraction=0.08):
    metrics = _template_metrics(template)
    return (
        0.75 * max(0.0, metrics['three_way_fraction'] - max_three_way_fraction)
        + 1.25 * max(0.0, metrics['four_way_fraction'] - max_four_way_fraction)
    )


def annealing_objective(score):
    return score[0] / (1 + score[1])


def annealing_objective_for_template(template, ks,
                                     max_three_way_fraction=0.4,
                                     max_four_way_fraction=0.08):
    base = annealing_objective(score_template(template, ks))
    collision = 0.2 * seed_collision_penalty(template)
    excess = recoverability_excess_penalty(
        template,
        max_three_way_fraction=max_three_way_fraction,
        max_four_way_fraction=max_four_way_fraction,
    )
    return base - collision - excess


def dominates(a: tuple, b: tuple) -> bool:
    return (a[0] >= b[0] and a[1] <= b[1]) and (a[0] > b[0] or a[1] < b[1])

def pareto_front(candidates: list) -> list:
    front = []
    for tpl, score in candidates:
        if not any(dominates(other_score, score) for _, other_score in candidates if other_score != score):
            front.append((tpl, score))
    return front


def _dedupe_candidates(candidates):
    unique = {}
    for tpl, score in candidates:
        if tpl not in unique:
            unique[tpl] = score
    return [(tpl, score) for tpl, score in unique.items()]


def _take_evenly_spaced(candidates, count, key):
    ordered = sorted(candidates, key=key)
    if count >= len(ordered):
        return ordered

    picks = []
    seen = set()
    indices = np.linspace(0, len(ordered) - 1, num=count)
    for raw_idx in indices:
        idx = int(round(float(raw_idx)))
        idx = max(0, min(idx, len(ordered) - 1))
        while idx in seen and idx + 1 < len(ordered):
            idx += 1
        while idx in seen and idx - 1 >= 0:
            idx -= 1
        if idx not in seen:
            seen.add(idx)
            picks.append(ordered[idx])
    return picks


def _select_annealing_seeds(candidates, num_designs, ks,
                            max_three_way_fraction=0.4,
                            max_four_way_fraction=0.08):
    unique_candidates = _dedupe_candidates(candidates)
    if num_designs >= len(unique_candidates):
        return unique_candidates

    selected = []
    selected_templates = set()

    frontier = pareto_front(unique_candidates)
    frontier = _take_evenly_spaced(frontier, min(num_designs, len(frontier)), key=lambda x: x[1][1])
    for tpl, score in frontier:
        selected.append((tpl, score))
        selected_templates.add(tpl)

    if len(selected) >= num_designs:
        return selected

    ranked = sorted(
        unique_candidates,
        key=lambda x: annealing_objective_for_template(
            x[0], ks,
            max_three_way_fraction=max_three_way_fraction,
            max_four_way_fraction=max_four_way_fraction,
        ),
        reverse=True,
    )

    pool = ranked[:min(len(ranked), max(num_designs * 4, num_designs))]
    pool = [candidate for candidate in pool if candidate[0] not in selected_templates]
    for tpl, score in _take_evenly_spaced(pool, min(num_designs - len(selected), len(pool)), key=lambda x: x[1][1]):
        selected.append((tpl, score))
        selected_templates.add(tpl)

    if len(selected) >= num_designs:
        return selected

    for tpl, score in ranked:
        if tpl in selected_templates:
            continue
        selected.append((tpl, score))
        selected_templates.add(tpl)
        if len(selected) >= num_designs:
            break

    return selected

def pick_elbow_candidate(pareto_candidates: list) -> tuple:
    if len(pareto_candidates) == 1:
        return pareto_candidates[0]
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
    _eps = 1e-12

    masks = _template_masks(template)
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


def _choice_weight(code):
    span = len(nuc_dict[code])
    if span <= 1:
        return 1.0
    if span == 2:
        return 2.0
    if span == 3:
        return 2.1
    return 1.6


def build_initial_template(length, max_homopolymer_len, no_gquad,
                           max_three_way_fraction=0.4,
                           max_four_way_fraction=0.08):
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
            weights = [_choice_weight(c) for c in choices]
            template += random.choices(choices, weights)[0]

        if len(template) == length:
            generating = False
        counter += 1
        if counter > 200:
            raise Exception('Could not generate templates, try relaxing recoverability thresholds or increasing max_homopolymer_len')
    return template

def mutate(template, max_homopolymer_len, iterations, no_gquad,
           max_three_way_fraction=0.4,
           max_four_way_fraction=0.08):
    # Mutate biased to increase entropy but prefer degenerate
    for _ in range(iterations):
        pos = random.randrange(len(template))
        alt_deg = [c for c in degenerate_keys if c != template[pos]]
        valid = [c for c in alt_deg
                 if not check_conditions(template[:pos] + c + template[pos+1:], max_homopolymer_len, no_gquad)]

        if not valid:
            continue
        weights = [_choice_weight(c) for c in valid]
        new_c = random.choices(valid, weights)[0]
        return template[:pos] + new_c + template[pos+1:]
    return template


def optimize_barcode_template(barcode_len, ks, initial_designs, opt_frac, iterations=1000,
                              max_homopolymer_len=3, no_gquad=False, initial_temp=0.1,
                              max_three_way_fraction=0.4, max_four_way_fraction=0.08,
                              **kwargs):
    num_designs = int(initial_designs * opt_frac)
    if initial_designs > 1:
        num_designs = max(2, num_designs)
    num_designs = min(initial_designs, num_designs)
    print(f'Generating {num_designs} optimized barcodes ')

    temps = np.linspace(start=initial_temp, stop=0.000001, num=iterations)
    designs = []
    for num in range(initial_designs):
        current = build_initial_template(
            barcode_len,
            max_homopolymer_len,
            no_gquad,
            max_three_way_fraction=max_three_way_fraction,
            max_four_way_fraction=max_four_way_fraction,
        )
        current_score = score_template(current, ks)
        designs.append([current, current_score])

    filtered_candidates = sorted(
        _select_annealing_seeds(
            designs,
            num_designs,
            ks,
            max_three_way_fraction=max_three_way_fraction,
            max_four_way_fraction=max_four_way_fraction,
        ),
        key=lambda x: annealing_objective_for_template(
            x[0], ks,
            max_three_way_fraction=max_three_way_fraction,
            max_four_way_fraction=max_four_way_fraction,
        ),
        reverse=True,
    )

    best_candidates = []
    for d in filtered_candidates:
        d = d[0]
        current_score = score_template(d, ks)
        current_objective = annealing_objective_for_template(
            d, ks,
            max_three_way_fraction=max_three_way_fraction,
            max_four_way_fraction=max_four_way_fraction,
        )
        temp = initial_temp

        for i in range(iterations):
            proposal = mutate(
                d,
                max_homopolymer_len,
                iterations,
                no_gquad,
                max_three_way_fraction=max_three_way_fraction,
                max_four_way_fraction=max_four_way_fraction,
            )
            prop_score = current_score if proposal == d else score_template(proposal, ks)
            proposal_objective = current_objective if proposal == d else annealing_objective_for_template(
                proposal, ks,
                max_three_way_fraction=max_three_way_fraction,
                max_four_way_fraction=max_four_way_fraction,
            )
            delta = proposal_objective - current_objective

            if delta >= 0 or random.random() < math.exp(delta / temp):
                d, current_score = proposal, prop_score
                current_objective = proposal_objective
            temp = temps[i]

        best_candidates.append((d, current_score))

    return best_candidates


def cli():
    # Optimize template barcodes to have diverse motifs (ks) and large combinatorial space
    parser = argparse.ArgumentParser(
        description="Generate a degenerate barcode template by simulated "
                    "annealing."
    )

    # Parameters if generating new barcode
    parser.add_argument('--barcode_len', type=int, default=75,
                        help='Length of the barcode template to generate.')
    parser.add_argument('--max_homopolymer_len', type=int, default=3,
                        help='Reject templates with possible homopolymers '
                             'longer than this.')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Simulated annealing iterations for each selected '
                             'candidate template.')
    parser.add_argument('--ks', type=int, default=[2,3,4,5], nargs='+',
                        help='Window sizes used to score repetitiveness and '
                             'diversity. The defaults work for most projects.')
    parser.add_argument('--initial_designs', type=int, default=100,
                        help='Number of random starting templates to generate '
                             'before selecting the best candidates for '
                             'annealing.')
    parser.add_argument('--opt_frac', type=float, default=0.5,
                        help='Fraction of initial templates to carry into the '
                             'annealing stage.')
    parser.add_argument('--no_gquad', default=False, action='store_true',
                        help='Avoid candidate templates with possible G-quad '
                             'motifs. Mostly useful for RNA barcodes.')
    parser.add_argument('--allow_fixed_bases', default=False, action='store_true',
                        help='Allow templates that contain fixed (non-degenerate) bases. '
                             'By default, only fully degenerate templates are returned.')
    parser.add_argument('--max_three_way_fraction', type=float, default=0.25,
                        help='Soft target for the fraction of 3-way degenerate '
                             'positions (B/D/H/V). Higher fractions are '
                             'penalized during optimization.')
    parser.add_argument('--max_four_way_fraction', type=float, default=0.05,
                        help='Soft target for the fraction of 4-way degenerate '
                             'positions (N). Higher fractions are penalized '
                             'during optimization.')

    args = parser.parse_args()

    candidates = optimize_barcode_template(**vars(args))

    # Diagnostics
    print(f"\n  Optimization returned {len(candidates)} candidates")

    pareto_candidates = pareto_front(candidates)
    print(f"  Pareto front: {len(pareto_candidates)} candidates")

    best_candidates = sorted(
        pareto_candidates,
        key=lambda x: annealing_objective_for_template(
            x[0],
            args.ks,
            max_three_way_fraction=args.max_three_way_fraction,
            max_four_way_fraction=args.max_four_way_fraction,
        ),
        reverse=True,
    )

    # Filter to fully-degenerate templates (no fixed A/C/G/T) unless --allow_fixed_bases
    if args.allow_fixed_bases:
        filtered_candidates = [(i[0], (i[1][0], i[1][1])) for i in best_candidates]
    else:
        filtered_candidates = [(i[0], (i[1][0], i[1][1]))
                               for i in best_candidates
                               if all(n not in i[0] for n in nucleotides)]

    print(f"  After filtering: {len(filtered_candidates)} candidates")
    unique_plot_points = len({
        (round(float(score[1]), 8), round(float(score[0]), 8))
        for _, score in filtered_candidates
    })
    print(f"  Unique plotted points: {unique_plot_points}")

    if not filtered_candidates:
        print("\n  WARNING: No candidates passed filtering!")
        print("  This can happen when ks values make it hard to find fully-degenerate templates.")
        print("  Try: --allow_fixed_bases, or increase --initial_designs, or adjust --ks")

        # Fall back to all pareto candidates so we at least show something
        if best_candidates:
            print(f"  Falling back to all {len(best_candidates)} pareto candidates for plotting\n")
            filtered_candidates = [(i[0], (i[1][0], i[1][1])) for i in best_candidates]
        else:
            print("  No candidates at all — check parameters.")
            return

    x = [i[1][1] for i in filtered_candidates]
    y = [i[1][0] for i in filtered_candidates]

    if plt is not None:
        fig, ax = plt.subplots()
        if sns is not None:
            sns.scatterplot(x=x, y=y, ax=ax)
        else:
            ax.scatter(x, y)
        ax.set_xlabel('Ambiguity penalty (minimize)')
        ax.set_ylabel('Mean Degeneracy (maximize)')
        ax.set_title('Barcode Template Pareto Front')

        plot_fn = 'barcode_pareto_front.png'
        fig.savefig(plot_fn, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved to {plot_fn}")

        try:
            plt.show()
        except Exception:
            pass

    elbow_candidate = pick_elbow_candidate(filtered_candidates)
    print(f'Optimized degenerate barcode template: {elbow_candidate[0]}')
    print(f'Mean degeneracy: {elbow_candidate[1][0]:.3f} / 4')
    print(f'Ambiguity penalty: {elbow_candidate[1][1]:.3f}')
    print(f'Motif penalty: {expanded_motif_penalty(elbow_candidate[0], tuple(args.ks)):.3f}')
    print(f'Seed collision penalty: {seed_collision_penalty(elbow_candidate[0]):.3f}')
    print(f'Recoverability penalty: {recoverability_penalty(elbow_candidate[0]):.3f}')
    print(f'Shift similarity: {shift_similarity_penalty(elbow_candidate[0]):.3f}')
    print(f'RC similarity: {reverse_complement_similarity_penalty(elbow_candidate[0]):.3f}')
    # print(f'  Score: penalty={elbow_candidate[1][1]:.4f}, degeneracy={elbow_candidate[1][0]:.4f}')

if __name__ == '__main__':
    cli()
