import yaml
import os
import plot_color as c


default_config = {
    'figure': {
        'figsize': [15, 8],
        'transparent': False,
        'path': None,       # f'first_hit_cross_{config}.png'
        'preview': True,
        'title': None,
        'legend': 'upper left'
    },
    'hosts': {}
}

def _get_default_config():
    return default_config.copy()


def read_config(path):
    config = _get_default_config()

    with open(path, 'r') as stream:
        user_config = yaml.safe_load(stream)
        config.update(user_config)

    for host in config['hosts']:
        if config['hosts'][host]['params'] == 'Auto':
            config['hosts'][host]['params'] = None

    if config['figure']['transparent']:
        config['figure']['facecolor'] = c.transparent
    else:
        config['figure']['facecolor'] = c.facecolor

    if config['figure']['path'] == 'Auto':
        fname = os.path.basename(path).rsplit('.', 1)[0]
        config['figure']['path'] = f'first_hit_cross_{fname}.png'


    return config
