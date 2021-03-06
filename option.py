import argparse


def getTrainParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_A', type=str, help="input dir of train dataset", required=True)
    parser.add_argument('--train_B', type=str, help="target dir of train dataset", required=True)
    parser.add_argument('--train_C', type=str, help="initial mask dir of train dataset", required=True)
    parser.add_argument('--valid_A', type=str, help="input dir of valid dataset", required=True)
    parser.add_argument('--valid_B', type=str, help="target dir of valid dataset", required=True)
    parser.add_argument('--valid_C', type=str, help="Initial mask dir of valid dataset", required=True)
    parser.add_argument('--batch_size', '-b', type=int, help="batch size of training", default=1)
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--loss_type', type=str, choices={"MS_CE", "MSP_CE"}, default="CE")
    parser.add_argument('--edge_mask_initial', type=float, nargs="*")
    parser.add_argument('--optimizer', type=str, choices={"Adam"}, default="Adam")
    parser.add_argument('--epoch_num', type=int, default=200)
    parser.add_argument('--class_num', '-c', type=int, required=True)
    parser.add_argument('--backbone_type', type=str)
    parser.add_argument('--reunit_type',type=str, nargs='*')
    parser.add_argument('--reunit_skipconnect', type=int, nargs='*', choices={0, 1})
    parser.add_argument('--dropout_probs', type=float, nargs='*')
    parser.add_argument('--recurrent_time', type=int)
    parser.add_argument('--loss_weights', type=float, nargs='+', required=True)
    parser.add_argument('--cross_validate', action="store_true")
    parser.add_argument('--cross_interval', type=int, default=200)
    parser.add_argument('--lr_manage_type', type=str, default="scheduler", choices={"stage", "log"})
    parser.add_argument('--epoch_stages', type=int, nargs='+')
    parser.add_argument('--lr_stages', type=float, nargs='+')
    parser.add_argument('--vutimes', type=int, default=20)
    parser.add_argument('--lr_re_rate', type=float, default=0.9)
    parser.add_argument('--min_lr', type=float)
    parser.add_argument('--max_lr', type=float)
    parser.add_argument('--outlayer_type', type=str, default="4")
    parser.add_argument("--best_start", type=int, default=0)
    parser.add_argument("--CE_Thre", type=float, nargs="+")
    parser.add_argument("--save_dir", type=str, default="checkpoint")
    return parser


def getPredictParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_dir", type=str, default="./submit")
    parser.add_argument("-m", "--merge_img", action="store_true")
    parser.add_argument("-w", "--weight_path", type=str)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--backbone_type', type=str)
    parser.add_argument('--reunit_type', type=str, nargs='*')
    parser.add_argument('--reunit_skipconnect', type=int, nargs='*', choices={0, 1})
    parser.add_argument('--dropout_probs', type=float, nargs='*',default=[1])
    parser.add_argument('--recurrent_time', type=int)
    parser.add_argument("-i", "--image_dir", type=str)
    parser.add_argument("-t", "--target_dir", type=str)
    parser.add_argument("-c", "--class_num", type=int, required=True)
    parser.add_argument("--recur_time", type=int, default=-1)
    parser.add_argument('--mode', type=str, default="single", choices={"single", "average"})
    parser.add_argument('--padOrCrop', type=str, default="pad", choices={"pad", "crop"})
    parser.add_argument('--outlayer_type', type=str, default="1")
    parser.add_argument("--save_dir", type=str, default="checkpoint")
    parser.add_argument('--gpu', action='store_true')
    return parser


