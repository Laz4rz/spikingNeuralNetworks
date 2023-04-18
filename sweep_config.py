sweep_config_SNN = {
    'method': 'grid',
    'name': 'bRuTeFoRcE_V2_oneLayer_ANN',
    'metric': {
        'goal': 'minimize',
        'name': 'train loss'
    },
    "description": "Lineaer(2, 2), Leaky()",
    'parameters': {
        'learning_rate': {
            'values': [0.01, 0.1]
        },
        'batch_size': {
            'values': [16, 32, 128]
        },
        'beta': {
            'values': [0.5, 0.9]
        },
        'threshold': {
            'values': [0.5, 0.9]
        },
        'timesteps': {
            'values': [5, 10]
        },
        'epochs': {
            'values': [30]
        },
        'rates': {
            'values': [9, 7]
        },
        "surrogate": {
            "values": ["fast_sigmoid", "sigmoid", "straight_through_estimator"] # "triangular" doesnt work for some reason, spike rate works bad
         },
        'seed': {
            'values': [1, 2137, 69]
        }
    }
}

sweep_config_ANN = {
    'method': 'grid',
    'name': 'bRuTeFoRcE_V2_ANN',
    'metric': {
        'goal': 'minimize',
        'name': 'train loss'
    },
    "description": "Linear(2, 8), Linear(8, 1), Sigmoid()",
    'parameters': {
        'learning_rate': {
            'values': [0.01, 0.1]
        },
        'batch_size': {
            'values': [16, 32, 128]
        },
        'epochs': {
            'values': [30]
        },
        'seed': {
            'values': [1, 2137, 69]
        }
    }
}