from smtpd import usage
import torch
from nlp_metrics.cocval_evalution import eval_nlp_scores
from preprocessing.pre_processing_msr import Msr_dataset
from preprocessing.pre_processing_msvd import Msvd_dataset
import sys
from model.Swin_BERT_Semantics import Swin_BERT_Semantics
from model.Swin_BERT import Swin_BERT
import getopt


def main():
    config = 'config/swin_base_bert.py'
    checkpoint_encoder = 'checkpoint/swin/swin_base_patch244_window877_kinetics400_22k.pth'
    lr = 0.00001

    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'a:d:s:b',
                                      longopts=['afs=', 'dataset=', 'semantics=','bestmodel='])
    except getopt.GetoptError as err:
        print(err)
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
                ckp_semantics = ''
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
        if o == '--bestmodel':
            path_model = v

    total_text_refrence = dataset.read_test()

    model.setup("0")
    test_loader=model.test_dataloader()
    #import pdb;
    #pdb.set_trace()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(path_model, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.freeze()

    total_text_generated = []

    for image, tag, caption in test_loader:
        image = image.to(device)
        model = model.to(device)
        caption = caption.to(device)

        output = model(image, caption)

        generate_text = output.cpu().numpy().tolist()

        generate_converted = model.tokenizer.batch_decode(generate_text, skip_special_tokens=True)

        for di in range(generate_converted.__len__()):
            print(generate_converted[di])
            total_text_generated.append([generate_converted[di]])

    metrics_dict = eval_nlp_scores(total_text_generated, total_text_refrence)
    print(metrics_dict['Bleu_1'][0])
    print(metrics_dict['Bleu_2'][0])
    print(metrics_dict['Bleu_3'][0])
    print(metrics_dict['Bleu_4'][0])
    print(metrics_dict['METEOR'][0])
    print(metrics_dict['CIDEr'][0])
    print(metrics_dict['ROUGE_L'][0])

if __name__ == '__main__':
    main()
