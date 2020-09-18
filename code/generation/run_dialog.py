import torch
import random
from model.utils import load_openai_weights_chinese, set_seed, f1_score
from model.transformer_model import TransformerModel
from model.text import myVocab
from config import get_model_config_dialog, get_test_config_dialog
import readline


def main():
    model_config = get_model_config_dialog()
    test_config = get_test_config_dialog()

    set_seed(test_config.seed)
    device = torch.device(test_config.device)

    vocab = myVocab(model_config.vocab_path)

    transformer = TransformerModel(n_layers=model_config.n_layers,
                                   n_embeddings=len(vocab),
                                   n_pos_embeddings=model_config.n_pos_embeddings,
                                   embeddings_size=model_config.embeddings_size,
                                   padding_idx=vocab.pad_id,
                                   n_heads=model_config.n_heads,
                                   dropout=model_config.dropout,
                                   embed_dropout=model_config.embed_dropout,
                                   attn_dropout=model_config.attn_dropout,
                                   ff_dropout=model_config.ff_dropout,
                                   bos_id=vocab.bos_id,
                                   eos_id=vocab.eos_id,
                                   max_seq_len=model_config.max_seq_len,
                                   beam_size=model_config.beam_size,
                                   length_penalty=model_config.length_penalty,
                                   n_segments=model_config.n_segments,
                                   annealing_topk=model_config.annealing_topk,
                                   annealing=model_config.annealing,
                                   diversity_coef=model_config.diversity_coef,
                                   diversity_groups=model_config.diversity_groups)

    transformer = transformer.to(device)
    state_dict = torch.load(test_config.last_checkpoint_path, map_location=device)
    temp = dict(state_dict['model'])
    keys = list(temp.keys())
    for key in keys:
        # new_key = '.'.join([i for i in key.split('.') if i != 'module'])
        new_key = key.replace('.module', '')
        temp[new_key] = temp.pop(key)
    transformer.load_state_dict(temp)
    transformer.eval()
    print('Weights loaded from {}'.format(test_config.last_checkpoint_path))


    def answer(message):
        message = ' '.join(message)
        message = vocab.string2ids(message)
        message = [vocab.bos_id] + message + [vocab.eos_id]
        message = message[:60]
        # print(message)
        contexts = [torch.tensor([c], dtype=torch.long, device=device) for c in [message] if len(c) > 0]
        prediction = transformer.predict(contexts)[0]
        prediction_str = vocab.ids2string(prediction)
        return prediction_str

    def answer_beams(message):
        message = ' '.join(message)
        message = vocab.string2ids(message)
        message = [vocab.bos_id] + message + [vocab.eos_id]
        message = message[:30]
        # print(message)
        contexts = [torch.tensor([c], dtype=torch.long, device=device) for c in [message] if len(c) > 0]
        predictions = transformer.predict_beams(contexts)[0]
        prediction_strs = [vocab.ids2string(prediction) for prediction in predictions]
        return prediction_strs

    '''
    with open('data/test200_output_noinit_noweight.txt', 'w', encoding='utf8') as fw:
        with open('data/test200.txt', 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            for line in lines:
                post, response = line.strip('\n').replace(' ', '').split('\t')
                ans = answer(post)
                fw.write('source:' + post + '\t' + 'target:' + response + '\t' + 'answer:' + ans + '\n')
    '''
    '''
    while True:
        message = input('>')
        ans = answer(message)
        print(ans)
    '''

    while True:
        message = input('>')
        ans = answer_beams(message)
        for i in ans:
            print(i)


if __name__ == '__main__':
    main()