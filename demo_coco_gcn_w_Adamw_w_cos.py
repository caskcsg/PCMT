import argparse
from engine_remix_coco import *

#from loss_fpngcn import *
from test_pcmt_coco import *
#from no_udrl_vis import *
from coco import *
from util import *
import os




#  python demo_coco_gcn_w_Adamw_w_cos.py coco -b 24 -i 384 --lr 3e-5 > log/demo_coco_gcn_w_AdamW_w_cos_202311141638.log  # Tmax=5000 lr=3e-5
#  python demo_coco_gcn_w_Adamw_w_cos.py coco -b 24 -i 384 --lr 3e-6 --resume checkpoint/coco/model_best_88.5145.pth.tar > log/demo_coco_gcn_resume_w_SGD_w_cos_202311151122.log
#  python demo_coco_gcn_w_Adamw_w_cos.py coco -b 24 -i 384 --lr 3e-5 --resume checkpoint/coco/model_best_88.5915.pth.tar > log/demo_coco_gcn_resume_w_SGD_wo_cos_202311151835.log  # wo cos scheduler w SGD
#  python demo_coco_gcn_w_Adamw_w_cos.py coco -b 24 -i 384 --lr 3e-5 --resume checkpoint/coco/model_best_88.5915.pth.tar > log/demo_coco_gcn_resume_w_AdamW_wo_cos_202311160914.log  # wo cos scheduler lr=3e-6 w Adamw
#  python demo_coco_gcn_w_Adamw_w_cos.py coco -b 24 -i 384 --lr 1e-5 --resume checkpoint/coco/model_best_88.5915.pth.tar > log/demo_coco_gcn_resume_w_AdamW_w_cos_202311161418.log  # w cos scheduler lr=1e-5 w Adamw






# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'


parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('data', metavar='DIR', default='coco',
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=384, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=54, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-s','--save', default='checkpoint/coco', type=str)
parser.add_argument('--cfg',
                        default='./CvT/experiments/imagenet/cvt/cvt-21-384x384.yaml',
                        help='experiment configure file name',
                        # required=True,
                        type=str)

def main_coco():
    global args, best_prec1, use_gpu
    args = parser.parse_args()
    # update_config(config, args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_gpu = torch.cuda.is_available()

    train_dataset = COCO2014(args.data, phase='train', inp_name='coco/coco_glove_word2vec.pkl')
    val_dataset = COCO2014(args.data, phase='val', inp_name='coco/coco_glove_word2vec.pkl')
    num_classes = 80

    # model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='/home/user/yza/ML-GCN/data/coco/coco_adj.pkl')
    model = gcn_resnet101(num_classes=num_classes, t=0.4,
                    adj_file='coco/coco_adj.pkl')  # ,dim=args.dim)
    # model = torch.nn.DataParallel(model)
    # model.to(device)
    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()

    # define optimizer
    # optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
    #                             lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr)  # @@@@@@@@@@@@@@@@@@@@@@@@@@1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8000)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
            'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes,'save_model_path':args.save}
    state['difficult_examples'] = True
    #state['save_model_path'] = 'checkpoint/coco/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    state['device'] = device
    # state['device_ids'] = args.device_ids
    if args.evaluate:
        state['evaluate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer, scheduler)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    main_coco()
