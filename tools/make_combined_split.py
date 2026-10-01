"""Build the split for training on all four EvoStruggle activities together.

The four within-activity splits (<Activity>_sepattempt.json) are merged:
the training videos of all five attempts form the `train` subset and the
validation videos form the `validation` subset. Video keys are prefixed with
the activity name (`<Activity>-<video>`), as in the activity-level
generalization splits of EvoStruggle.

Usage (from the root of the repository):
    python tools/make_combined_split.py [--evostruggle data/EvoStruggle] [--out splits/combined_split.json]
"""
import argparse
import json
import os

ACTIVITIES = ['Origami', 'Shuffle_Cards', 'Tangram', 'Tying_Knots']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--evostruggle', default='data/EvoStruggle', help='root of the EvoStruggle repository')
    parser.add_argument('--out', default='splits/combined_split.json')
    args = parser.parse_args()

    database = {}
    for act in ACTIVITIES:
        path = os.path.join(args.evostruggle, 'splits', 'separate_attempts', act, f'{act}_sepattempt.json')
        with open(path) as f:
            db = json.load(f)['database']
        for video, info in db.items():
            subset = info['subset'].lower()
            if subset.startswith('train'):
                subset = 'train'
            elif subset != 'validation':
                raise ValueError(f'unexpected subset {subset} in {path}')
            database[f'{act}-{video}'] = dict(info, subset=subset)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'version': 'combined', 'database': database}, f, indent=4)
    counts = {}
    for v in database.values():
        counts[v['subset']] = counts.get(v['subset'], 0) + 1
    print(f'Wrote {args.out}: {counts}')


if __name__ == '__main__':
    main()
