from smtpd import usage
import pytorch_lightning as pl
from preprocessing.pre_processing_msr import Msr_dataset
from preprocessing.pre_processing_msvd import Msvd_dataset
import sys
from model.Swin_BERT_Semantics import Swin_BERT_Semantics
from model.Swin_BERT import Swin_BERT
import getopt


def main():
    config = 'config/swin_base_bert.py'
    checkpoint_encoder = 'checkpoint/swin/swin_base_patch244_window877_kinetics400_22k.pth'
    checkpoint_path = ''
    lr = 0.00001
    gra_clip = 0.05

    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'a:s:d:c',
                                      longopts=['afs=', 'dataset=', 'semantics=', 'ckp_semantics='])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)

    for o, v in optlist:
        if o == '--afs':
            if v == "True":
                adaptive = True
            else:
                adaptive = False

        if o == '--dataset':
            if v == 'msrvtt':
                dataset = Msr_dataset("data/MSRVTT")
            if v == 'msvd':
                dataset = Msvd_dataset("data/MSVD")
        if o == '--semantics':

            if v == "True":
                ckp_semantics = optlist[3][1]
                model = Swin_BERT_Semantics(mlp_freeze=False, swin_freeze=True,
                                            in_size=1024, lr=lr, lambda_=0.1, hidden_sizes=[2048, 1024], out_size=768,
                                            gra_clip=0, drop_swin=0, weight_decay=0, max_length=20,
                                            drop_mlp=0.1, drop_bert=0.3, bs=2, dataset=dataset,
                                            config_data=config, check_semantics=ckp_semantics,
                                            checkpoint_encoder=checkpoint_encoder, using_adaptive=adaptive)
            else:
                model = Swin_BERT(swin_freeze=True,
                                  lr=lr, gra_clip=0, drop_swin=0, weight_decay=0, max_length=20,
                                  drop_bert=0.3, bs=2, dataset=dataset, config_data=config,
                                  checkpoint_encoder=checkpoint_encoder, using_adaptive=adaptive)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="checkpoint-{epoch:02d}-{harmonic_mean_bleu_meteor:.3e}",
        monitor="harmonic_mean_bleu_meteor",
        save_top_k=10,
        mode="max",
    )
    trainer = pl.trainer.Trainer(gpus=4, accelerator='ddp', gradient_clip_val=gra_clip,
                                 precision=16, default_root_dir=checkpoint_path,
                                 callbacks=[checkpoint_callback])
    trainer.fit(model)


if __name__ == '__main__':
    main()
