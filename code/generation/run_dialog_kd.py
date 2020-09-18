from run_dialog_interface import dialog
from metrics import calc_f1, calc_bleu, calc_distinct, calc_avg_len

bz1 = 5; lp1 = 1.6
bz2 = 5; lp2 = 1.0
bz3 = 5; lp3 = 2.0
bz4 = 5; lp4 = 1.6
bz5 = 5; lp5 = 1.6

crowded = dialog('/root/generation_with_augmentation/checkpoints/dialog_v2/crowded/last_checkpoint15', bz1, lp1)
fake = dialog('/root/generation_with_augmentation/checkpoints/dialog_fake_v2/last_checkpoint7', bz2, lp2)
fake_kd = dialog('/root/generation_with_augmentation/checkpoints/dialog_kd/fake/last_checkpoint28', bz3, lp3)
crowded_fake = dialog('/root/generation_with_augmentation/checkpoints/dialog_v2/crowded_fake/last_checkpoint18', bz4, lp4)
crowded_fake_kd = dialog('/root/generation_with_augmentation/checkpoints/dialog_kd/crowded_fake/last_checkpoint29', bz5, lp5)


crowded_pairs = []
fake_pairs = []
fake_kd_pairs = []
crowded_fake_pairs = []
crowded_fake_kd_pairs = []

if __name__ == '__main__':
    with open('dataset/dialog/test_1k.txt', 'r', encoding='utf8') as fr:
        lines = fr.readlines()
        cnt = 0
        with open('dataset/dialog/test_1k_output_kd_bm5_lp_diff_v2.txt', 'w', encoding='utf8') as fw:
            fw.write('question\tanswer\tcrowded\tfake\tfake_kd\tcrowded_fake\tcrowded_fake_kd\n')
            for line in lines:
                cnt += 1
                if cnt%100==0:
                    print(cnt)
                q,a = line.strip('\n').split('\t')
                a1 = crowded.answer_beams(q)
                crowded_pairs.append([list(a1), list(a)])
                a2 = fake.answer_beams(q)
                fake_pairs.append([list(a2), list(a)])
                a3 = fake_kd.answer_beams(q)
                fake_kd_pairs.append([list(a3), list(a)])
                a4 = crowded_fake.answer_beams(q)
                crowded_fake_pairs.append([list(a4), list(a)])
                a5 = crowded_fake_kd.answer_beams(q)
                crowded_fake_kd_pairs.append([list(a5), list(a)])
                res = [q, a, a1, a2, a3, a4, a5]
                fw.write('\t'.join(res) + '\n')
        for res in [crowded_pairs, fake_pairs, fake_kd_pairs, crowded_fake_pairs, crowded_fake_kd_pairs]:
            f1 = calc_f1(res)
            bleu = calc_bleu(res)
            distinct = calc_distinct(res)
            avg_len = calc_avg_len(res)
            print('f1: ', f1)
            print('bleu: ', bleu)
            print('distinct: ', distinct)
            print('avg_len: ', avg_len)
    '''
    while True:
        message = input('>')
        print('crowded', crowded.answer_beams(message))
        print('crowded_fake', crowded_fake.answer_beams(message))
        print('fake', fake.answer_beams(message))
    '''