'''
python verification_tsv.py /modelblob/users/v-chengw/data/librilight/infer_data/phn/dev_clean_wav_3to10sec.tsv  /modelblob/users/v-chengw/models/speech_lm/nat_adaln_nocond_step_1_8_crop750_bs18000_lr5e-4_400k/sample_last_phn/clean/train.tsv wavlm_large wavlm_large_finetune.pth /modelblob/users/v-chengw/models/speech_lm/nat_adaln_nocond_step_1_8_crop750_bs18000_lr5e-4_400k/sample_last_phn/clean/train_asv_scores_with_wavlm_large_finetune.txt
'''
import tqdm
import argparse
from verification import verification

parser = argparse.ArgumentParser()
parser.add_argument('tsv1')
parser.add_argument('tsv2')
parser.add_argument('model_name')
parser.add_argument('checkpoint')
parser.add_argument('scores')
args = parser.parse_args()

tsv1 = open(args.tsv1)
tsv1_root = tsv1.readline().strip()
tsv1 = tsv1.readlines()

tsv2 = open(args.tsv2)
tsv2_root = tsv2.readline().strip()
tsv2 = tsv2.readlines()

scores_w = open(args.scores, 'w')

assert len(tsv1) == len(tsv2)

model = None
for t1, t2 in tqdm.tqdm(zip(tsv1, tsv2), total=len(tsv1)):
    t1_name = t1.split()[0]
    t2_name = t2.split()[0]
    sim, model = verification(args.model_name,  tsv1_root+'/'+t1_name, tsv2_root+'/'+t2_name, use_gpu=True, checkpoint=args.checkpoint, wav1_start_sr=0, wav2_start_sr=0, wav1_end_sr=None, wav2_end_sr=None, model=model)
    scores_w.write(f'{t1_name}|{t2_name}\t{sim.cpu().item()}\n')
    scores_w.flush()

