#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

from torch.utils.data import Dataset
import random

class S2sDataset_dialog(Dataset):
    def __init__(self, paths, vocab, max_lengths=2048):
        if isinstance(paths, str):
            print('path is str')
            paths = [paths]

        self.vocab = vocab
        self.max_lengths = max_lengths
        self.data = S2sDataset_dialog.make_dataset(paths, vocab, max_lengths)
        print(len(self.data))

    @staticmethod
    def make_dataset(paths, vocab, max_lengths):
        dataset = []
        for path in paths:
            with open(path, 'r', encoding='utf8') as fr:
                lines = fr.readlines()
                for line in lines:
                    context, response = line.strip('\n').split('\t')
                    context = vocab.string2ids(' '.join(context))
                    response = vocab.string2ids(' '.join(response))
                    dataset.append((context, response))
        random.shuffle(dataset)
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, response = self.data[idx]
        context = [self.vocab.bos_id] + context + [self.vocab.eos_id]
        response = [self.vocab.bos_id] + response + [self.vocab.eos_id]
        context = context[-40:]
        response = response[:40]
        return context, response

class S2sDataset_dialog_overlap(Dataset):
    def __init__(self, paths, vocab, max_lengths=2048):
        if isinstance(paths, str):
            print('path is str')
            paths = [paths]

        self.vocab = vocab
        self.max_lengths = max_lengths
        self.data = S2sDataset_dialog_overlap.make_dataset(paths, vocab, max_lengths)
        print(len(self.data))
        print(self.data[0])

    @staticmethod
    def make_dataset(paths, vocab, max_lengths):
        dataset = []
        for path in paths:
            with open(path, 'r', encoding='utf8') as fr:
                lines = fr.readlines()
                for line in lines:
                    context, response, overlap = line.strip('\n').split('\t')
                    context = vocab.string2ids(' '.join(context))
                    response = vocab.string2ids(' '.join(response))
                    overlap = int(overlap)
                    if overlap < 2:
                        overlap_sym = vocab.token2id['<M0>']
                    elif overlap < 4:
                        overlap_sym = vocab.token2id['<M1>']
                    elif overlap < 6:
                        overlap_sym = vocab.token2id['<M2>']
                    else:
                        overlap_sym = vocab.token2id['<M3>']
                    dataset.append((context, response, overlap_sym))
        random.shuffle(dataset)
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, response, overlap = self.data[idx]
        context = [self.vocab.bos_id] + context[-37:] + [overlap] + [self.vocab.eos_id]
        response = [self.vocab.bos_id] + response[-38:] + [self.vocab.eos_id]
        #context = context[-40:]
        #response = response[:40]
        return context, response

class S2sDataset_poem(Dataset):
    def __init__(self, paths, vocab, max_lengths=2048):
        if isinstance(paths, str):
            paths = [paths]

        self.vocab = vocab
        self.max_lengths = max_lengths
        self.data = S2sDataset_poem.make_dataset_wu(paths[0], vocab, max_lengths)

    @staticmethod
    def make_dataset_xiandai(paths, vocab, max_lengths):
        dataset = []
        with open(paths, 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            for line in lines:
                context, response = line.strip('\n').split('\t')
                context = vocab.string2ids(context)
                response = vocab.string2ids(response)
                dataset.append((context[:18], response[:118]))
        return dataset

    @staticmethod
    def make_dataset_wu(paths, vocab, max_lengths):
        dataset = []
        with open(paths, 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            for line in lines:
                target, source = line.strip('\n').split('\t')
                source = ' <p> '.join([' '.join(i) for i in source.split()])
                target = ' '.join(target)
                context = vocab.string2ids(source)
                response = vocab.string2ids(target)
                dataset.append((context[:18], response[:30]))
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, response = self.data[idx]
        context = [self.vocab.bos_id] + context + [self.vocab.eos_id]
        response = [self.vocab.bos_id] + response + [self.vocab.eos_id]
        # context = context[:20]
        # response = response[:120]
        return context, response

class S2sDataset_meme(Dataset):
    def __init__(self, paths, vocab, max_lengths=2048):
        if isinstance(paths, str):
            paths = [paths]

        self.vocab = vocab
        self.max_lengths = max_lengths
        self.data = S2sDataset_meme.make_dataset(paths[0], vocab, max_lengths)

    @staticmethod
    def make_dataset(paths, vocab, max_lengths):
        dataset = []
        with open(paths, 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            for line in lines:
                context, response = line.strip('\n').split('\t')
                context = vocab.string2ids(context)
                response = vocab.string2ids(response)
                dataset.append((context, response))
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, response = self.data[idx]
        context = [self.vocab.bos_id] + context + [self.vocab.eos_id]
        response = [self.vocab.bos_id] + response + [self.vocab.eos_id]
        context = context[:30]
        response = response[:30]
        return context, response