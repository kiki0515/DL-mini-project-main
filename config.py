def get_configs():
    train_params = {
        "device": "cuda",
        "max_epoch": 100,
        "output_dir": "/outputs/allAugs_AdamW_lr0.01_WeightDecay5e-4_batchSize128"
    }

    dataset_params = {
        "train_bs": 128,
        "test_bs": 256,
    }

    model_params = {

    }

    callback_params = {
        "patience": 5,
        "save_final_model": True,
    }

    optimizer_params = {
        "lr": 0.01,
        "type": "AdamW",
        "kwargs": {
            "weight_decay": 5e-4,
            "amsgrad": True
        }
    }

    return train_params, dataset_params, callback_params, optimizer_params
