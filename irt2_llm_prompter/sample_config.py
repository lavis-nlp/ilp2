import irt2
p_data = irt2.ENV.DIR.DATA
def p():
    print(p_data)

all_config = {
    'datasets': {
        'irt2-cde-tiny': {
            'path': p_data / 'irt2' / 'irt2-cde-tiny',
            'loader': 'irt2',
            'percentage': {
                'validation': 0.17,
                'test': 0.02,
            },
        },
        'irt2-cde-small': {
            'path': p_data / 'irt2' / 'irt2-cde-small',
            'loader': 'irt2',
            'percentage': {
                'validation': 0.08,
                'test': 0.02,
            },
        },
        'irt2-cde-medium': {
            'path': p_data / 'irt2' / 'irt2-cde-medium',
            'loader': 'irt2',
            'percentage': {
                'validation': 0.04,
                'test': 0.01,
            },
        },
        'irt2-cde-large': {
            'path': p_data / 'irt2' / 'irt2-cde-large',
            'loader': 'irt2',
            'percentage': {
                'validation': 0.05,
                'test': 0.02,
            },
        },
        'WN18RR': {
            'path': p_data/ 'blp' / 'WN18RR',
            'loader': 'blp/wn18rr',
            'percentage': {
                'validation': 0.06,
                'test': 0.06,
            },
        },
        'FB15k-237': {
            'path': p_data/ 'blp' / 'FB15k-237',
            'loader': 'blp/fb15k237',
            'percentage': {
                'validation': 0.03,
                'test': 0.03,
            },
        },
        # {
        #     'path': p_data/ 'blp' / 'Wikidata5M',
        #     'loader': 'blp/wikidata5m',
        #     'percentage': {
        #         'validation': 0.09,
        #         'test': 0.08,
        #     },
        # },
    },
    'models': [
        # 'true-vertices',
        'true-mentions',
        # 'random-guessing',
    ],
    'splits': [
        'validation',
        'test',
    ],
    'seed': 31189,
}