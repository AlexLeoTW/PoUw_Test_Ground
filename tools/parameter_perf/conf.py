plots = [
    {
        'col_name': 'acc_score(%)',     # can be any string
        # 'title': 'acc_score(%)',      # string, Default: col_name
        'scale': True,                  # True/ False, Default: False
        'annotate': '{:.2%}'             # True / False / format
    }, {
        'col_name': 'train_time',
        'scale': True,
        'annotate': '{:.1f}'
    }
]
