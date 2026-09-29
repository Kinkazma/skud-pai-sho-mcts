"""Exhaustive score-ranking audit of verified R1 branches; no training/runtime writes."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def audit(corpus, output):
    roots_path = corpus / 'census-03/roots.json'
    branches_path = corpus / 'branches-02/branches.jsonl'
    verification = json.loads((corpus / 'branches-02/verification.json').read_text())
    digest = hashlib.sha256(branches_path.read_bytes()).hexdigest()
    if digest != verification['branches_sha256']:
        raise ValueError('verified branches changed')
    roots = {r['key']: r for r in json.loads(roots_path.read_text())}
    rows = collections.defaultdict(list)
    with branches_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            rows[row['root']].append(row)
    if sum(map(len, rows.values())) != verification['verified_branches']:
        raise ValueError('incomplete verified population')
    if set(rows) != set(roots):
        raise ValueError('root coverage mismatch')
    totals = {s: collections.Counter() for s in ('train', 'heldout')}
    details = []
    for key, actions in sorted(rows.items()):
        root = roots[key]
        split = 'heldout' if root['held_out'] else 'train'
        tally = totals[split]
        tally['roots'] += 1
        tally['actions'] += len(actions)
        best = max(a['score_difference_mover'] for a in actions)
        maxima = [a for a in actions if a['score_difference_mover'] == best]
        wins = [a for a in actions if a['outcome'] == 'win']
        winning_maxima = [a for a in maxima if a['outcome'] == 'win']
        row = dict(root=key, group=root['group'], split=split,
                   psr=str(corpus / 'census-03' / root['psr']),
                   root_sha256=root['sha256'], legal=len(actions),
                   maximum_score_difference=best, maxima=len(maxima),
                   known_wins=len(wins), winning_maxima=len(winning_maxima))
        if wins:
            tally['roots_with_immediate_win'] += 1
            tally['score_excludes_all_wins'] += not winning_maxima
            tally['score_ties_win_and_nonwin'] += 0 < len(winning_maxima) < len(maxima)
            tally['all_score_maxima_win'] += len(winning_maxima) == len(maxima)
            # Descriptive expectation if ties were uniform, not measured play.
            tally['sum_uniform_tie_win_probability'] += len(winning_maxima) / len(maxima)
            if not winning_maxima:
                row['excluded_win_witness'] = wins[0]
                row['maximum_witness'] = maxima[0]
        comparable = [a for a in actions if a['threat']['status'] in ('absent', 'present')]
        safe = [a for a in comparable if a['threat']['status'] == 'absent']
        exposed = [a for a in comparable if a['threat']['status'] == 'present']
        if safe and exposed:
            tally['roots_with_defence_contrast'] += 1
            score = max(a['score_difference_mover'] for a in comparable)
            top = [a for a in comparable if a['score_difference_mover'] == score]
            bad = [a for a in top if a['threat']['status'] == 'present']
            unsafe = len(bad) == len(top)
            tally['score_excludes_all_one_decision_defences'] += unsafe
            tally['score_ties_defence_and_threat'] += 0 < len(bad) < len(top)
            tally['all_score_maxima_avoid_one_decision_threat'] += not bad
            row['defence_contrast'] = True
            row['all_maxima_threatened'] = unsafe
            if unsafe:
                row['safe_witness'] = safe[0]
                row['threatened_maximum_witness'] = bad[0]
        # Same-root equal score does not necessarily imply equal tactical labels.
        equal = collections.defaultdict(list)
        for a in actions:
            equal[tuple(a['score_after_host_guest'])].append(a)
        ambiguous = any(any(a['outcome'] == 'win' for a in group)
                        and any(a['outcome'] != 'win' for a in group)
                        for group in equal.values())
        tally['same_score_pair_win_vs_nonwin'] += ambiguous
        row['same_score_pair_win_vs_nonwin'] = ambiguous
        for a in actions:
            if a['ending'] == 'exhaustion':
                predicted = 'win' if a['score_difference_mover'] > 0 else (
                    'loss' if a['score_difference_mover'] < 0 else 'draw')
                if predicted != a['outcome']:
                    raise ValueError('exhaustion score sign inconsistent')
                tally['last_plant_' + predicted] += 1
        details.append(row)
    result = dict(totals=totals, details=details, branches_sha256=digest,
                  exhaustive_on_frozen_corpus=True, neural_learning_tested=False,
                  publication_tested=False, score_only_ranking_rejected=any(
                      t['score_excludes_all_wins'] or t['score_excludes_all_one_decision_defences']
                      for t in totals.values()))
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    audit(args.corpus, args.output)
