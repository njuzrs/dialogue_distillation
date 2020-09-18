from attrdict import AttrDict
from model.utils import openai_transformer_config


# transformer config
def get_model_config_dialog():
    default_config = openai_transformer_config()
    config = AttrDict({'vocab_path': '/root/generation_with_augmentation/parameters/vocab.txt',
                       'checkpoint_path': '/root/generation_with_augmentation/checkpoints/dialog_300k/crowded_extend_both_kd/last_checkpoint',
                       'n_layers': 12,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'max_seq_len': 32,
                       'beam_size': 1,
                       'diversity_coef': 0,
                       'diversity_groups': 1,
                       'temperature': 1.0,
                       'annealing_topk': None,
                       'annealing': 0,
                       'length_penalty': 1.0,
                       'n_segments': None})

    return config


def get_trainer_config_dialog():
    config = AttrDict({'n_epochs': 100,
                       'batch_size': 256,
                       'batch_split': 32,
                       'lr': 6.25e-5,
                       'lr_warmup': 1000,
                       'lm_weight': 0.5,
                       'risk_weight': 0,
                       'n_jobs': 4,
                       'label_smoothing': 0.1,
                       'clip_grad': None,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'load_last': True,
                       'openai_parameters_dir': '/root/generation_with_augmentation/parameters/chinese_pretrain.pt',
                       'last_checkpoint_path': '/root/generation_with_augmentation/checkpoints/dialog_300k/crowded_extend_both_kd/last_checkpoint',
                       'teacher_checkpoint_path': '/root/generation_with_augmentation/checkpoints/dialog_300k/crowded/last_checkpoint15',
                       'interrupt_checkpoint_path': '/root/generation_with_augmentation/checkpoints/dialog_300k/crowded_extend_both_kd/interrupt_checkpoint',
                       'train_datasets': [#'/root/generation_with_augmentation/dataset/dialog/crowded_500k.txt',
                                          #'/root/generation_with_augmentation/dataset/dialog/crowded_300k.txt',
                                          #'/root/generation_with_augmentation/dataset/dialog/crowded_100k.txt',
                                          #'/root/generation_with_augmentation/dataset/dialog/crowded_unpaired_500k.txt',
                                          #'/root/generation_with_augmentation/dataset/dialog/fake_unique.txt',
                                          #'/root/generation_with_augmentation/dataset/dialog/crowded_300k_eda_v2.txt',
                                          #'/root/generation_with_augmentation/dataset/dialog/crowded_300k_cvae_post.txt',
                                          #'/root/generation_with_augmentation/dataset/dialog/crowded_dirty_300k.txt',
                                          #'/root/generation_with_augmentation/dataset/dialog/crowded_300k_bt.txt',
                                          #'/root/generation_with_augmentation/dataset/dialog/crowded_300k_extend_post.txt',
                                          #'/root/generation_with_augmentation/dataset/dialog/crowded_300k_extend_resp.txt',
                                          '/root/generation_with_augmentation/dataset/dialog/crowded_300k_extend_both.txt',
                                          #'/root/generation_with_augmentation/dataset/dialog/fake_wo_matching_generation.txt',
                                          ],
                       'test_datasets': ['/root/generation_with_augmentation/dataset/dialog/valid_9k.txt']})
    return config


def get_test_config_dialog():
    config = AttrDict({'seed': 0,
                       'device': 'cuda',
                       'load_last': True,
                       'openai_parameters_dir': '/root/generation_with_augmentation/parameters/chinese_pretrain.pt',
                       'last_checkpoint_path': '/root/generation_with_augmentation/checkpoints/dialog_kd/crowded_fake/last_checkpoint11'})
    return config


# transformer config overlap
def get_model_config_dialog_overlap():
    default_config = openai_transformer_config()
    config = AttrDict({'vocab_path': '/root/generation_with_augmentation/parameters/vocab_overlap.txt',
                       'checkpoint_path': '/root/generation_with_augmentation/checkpoints/dialog_overlap/v1/last_checkpoint',
                       'n_layers': 12,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'max_seq_len': 32,
                       'beam_size': 5,
                       'diversity_coef': 0,
                       'diversity_groups': 1,
                       'temperature': 1.0,
                       'annealing_topk': None,
                       'annealing': 0,
                       'length_penalty': 1.5,
                       'n_segments': None})

    return config


def get_trainer_config_dialog_overlap():
    config = AttrDict({'n_epochs': 30,
                       'batch_size': 256,
                       'batch_split': 16,
                       'lr': 6.25e-5,
                       'lr_warmup': 1000,
                       'lm_weight': 0.5,
                       'risk_weight': 0,
                       'n_jobs': 4,
                       'label_smoothing': 0.1,
                       'clip_grad': None,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'load_last': False,
                       'openai_parameters_dir': '/root/generation_with_augmentation/parameters/chinese_pretrain.pt',
                       'last_checkpoint_path': '/root/generation_with_augmentation/checkpoints/dialog_overlap/v1/last_checkpoint',
                       'interrupt_checkpoint_path': '/root/generation_with_augmentation/checkpoints/dialog_overlap/v1/interrupt_checkpoint',
                       'train_datasets': ['/root/generation_with_augmentation/dataset/dialog/crowded_500k_overlap.txt',
                                          #'/root/generation_with_augmentation/dataset/dialog/crowded_unpaired_500k.txt',
                                          #'/root/generation_with_augmentation/dataset/dialog/fake_unique.txt',
                                          ],
                       'test_datasets': ['/root/generation_with_augmentation/dataset/dialog/valid_9k_overlap.txt']})
    return config

def get_test_config_dialog_overlap():
    config = AttrDict({'seed': 0,
                       'device': 'cuda',
                       'load_last': True,
                       'openai_parameters_dir': '/root/generation_with_augmentation/parameters/chinese_pretrain.pt',
                       'last_checkpoint_path': '/root/generation_with_augmentation/checkpoints/dialog_overlap/v1/last_checkpoint15'})
    return config

# transformer config
def get_model_config_poem():
    default_config = openai_transformer_config()
    config = AttrDict({'vocab_path': '/root/generation_with_augmentation/parameters/vocab.txt',
                       'checkpoint_path': '/root/generation_with_augmentation/checkpoints/poem_wu/last_checkpoint',
                       'n_layers': 12,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'max_seq_len': 128,
                       'beam_size': 15,
                       'diversity_coef': 0.5,
                       'diversity_groups': 5,
                       'temperature': 0.8,
                       'annealing_topk': 20,
                       'annealing': 1.0,
                       'length_penalty': 0.6,
                       'n_segments': None})

    return config


def get_trainer_config_poem():
    config = AttrDict({'n_epochs': 30,
                       'batch_size': 256,
                       'batch_split': 8,
                       'lr': 6.25e-5,
                       'lr_warmup': 1000,
                       'lm_weight': 0,
                       'risk_weight': 0,
                       'n_jobs': 4,
                       'label_smoothing': 0,
                       'clip_grad': None,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'load_last': False,
                       'openai_parameters_dir': '/root/generation_with_augmentation/parameters/chinese_pretrain.pt',
                       'last_checkpoint_path': '/root/generation_with_augmentation/checkpoints/poem_wu/last_checkpoint',
                       'interrupt_checkpoint_path': '/root/generation_with_augmentation/checkpoints/poem_wu/interrupt_checkpoint',
                       'train_datasets': ['/root/generation_with_augmentation/dataset/poem/train_wu.txt'],
                       'test_datasets': ['/root/generation_with_augmentation/dataset/poem/valid_wu.txt']})
    return config


def get_test_config_poem():
    config = AttrDict({'seed': 0,
                       'device': 'cuda',
                       'load_last': True,
                       'openai_parameters_dir': '/root/generation_with_augmentation/parameters/chinese_pretrain.pt',
                       'last_checkpoint_path': '/root/generation_with_augmentation/checkpoints/poem/last_checkpoint20'})

    return config

# transformer config
def get_model_config_meme():
    default_config = openai_transformer_config()
    config = AttrDict({'vocab_path': '/root/generation_with_augmentation/parameters/vocab.txt',
                       'checkpoint_path': '/root/generation_with_augmentation/checkpoints/meme_all/last_checkpoint',
                       'n_layers': 6,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'max_seq_len': 30,
                       'beam_size': 5,
                       'diversity_coef': 0,
                       'diversity_groups': 1,
                       'temperature': 1.0,
                       'annealing_topk': None,
                       'annealing': 0,
                       'length_penalty': 1.0,
                       'n_segments': None})

    return config


def get_trainer_config_meme():
    config = AttrDict({'n_epochs': 50,
                       'batch_size': 256,
                       'batch_split': 8,
                       'lr': 6.25e-5,
                       'lr_warmup': 1000,
                       'lm_weight': 0,
                       'risk_weight': 0,
                       'n_jobs': 4,
                       'label_smoothing': 0,
                       'clip_grad': None,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'load_last': False,
                       'openai_parameters_dir': '/root/generation_with_augmentation/parameters/chinese_pretrain.pt',
                       'last_checkpoint_path': '/root/generation_with_augmentation/checkpoints/meme_all/last_checkpoint',
                       'interrupt_checkpoint_path': '/root/generation_with_augmentation/checkpoints/meme_all/interrupt_checkpoint',
                       'train_datasets': ['/root/generation_with_augmentation/dataset/meme/train_all.txt'],
                       'test_datasets': ['/root/generation_with_augmentation/dataset/meme/valid_all.txt']})
    return config


def get_test_config_meme():
    config = AttrDict({'seed': 0,
                       'device': 'cuda',
                       'load_last': True,
                       'openai_parameters_dir': '/root/generation_with_augmentation/parameters/chinese_pretrain.pt',
                       'last_checkpoint_path': '/root/generation_with_augmentation/checkpoints/meme_all/last_checkpoint19'})
    return config