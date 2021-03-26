import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from sklearn.metrics import *
import cv2, glob, tqdm
from matplotlib import pyplot as plt
import fire


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def prob2mask(prob, threshold=0.5):
    prob[prob>threshold] = 1
    prob[prob<=threshold] = 0
    return prob.astype(np.uint8)


def metrics(gt_dir, pred_dir, prob_range=(0, 100), beta=1, out="./result.png"):
    gts = glob.glob(gt_dir)
    prob_preds = glob.glob(pred_dir)
    fbeta_score_logs = []
    accuracy_score_logs = []
    mIoU_log = []
    MAE = None
    AUC = None
    AP =None

    for the in range(prob_range[0], prob_range[1]):
        print("threshold=", the, "...")
        f_scores = []
        accuracies = []
        hist = np.zeros((2, 2))
        MAEs = []
        AUCs = []
        APs = []
        for i in tqdm.tqdm(range(len(gts))):
            gt = np.ravel(cv2.imread(gts[i])[:, :, 0])
            prob = np.ravel(cv2.imread(prob_preds[i])[:, :, 0]) / 255
            f_score = fbeta_score(gt, prob2mask(np.copy(prob), threshold=the / 100), beta=np.sqrt(beta))
            f_scores.append(f_score)
            if AUC is None:
                mae = mean_absolute_error(gt, prob)
                MAEs.append(mae)
                auc = roc_auc_score(gt, prob)
                AUCs.append(auc)
                ap = average_precision_score(gt, prob)
                APs.append(ap)
            hist += fast_hist(gt, prob2mask(np.copy(prob), threshold=the / 100), 2)
            accuracies.append(accuracy_score(gt, prob2mask(np.copy(prob), threshold=the / 100)))

        fbeta_score_logs.append(np.average(f_scores))
        accuracy_score_logs.append(np.average(accuracies))
        mIoU_classes = per_class_iu(hist)
        mIoU_log.append(np.average(mIoU_classes))
        if AUC is None:
            MAE = (np.mean(MAEs))
            AUC = (np.mean(AUCs))
            AP = (np.mean(APs))
        print("threshold: {:f} f_score: {:f} accuracy: {:f} accuracy_var: {:f} mIoU: {:f}".format(
            the,
            np.average(f_scores),
            np.average(accuracies),
            np.var(accuracies),
            np.average(mIoU_classes)
        ))

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(fbeta_score_logs)
    plt.title(r'$f_\beta-measure:$ {:f}'.format(np.max(fbeta_score_logs)))
    plt.subplot(4, 1, 2)
    plt.plot(accuracy_score_logs)
    plt.title('max accuracy:{:f}'.format(np.max(accuracy_score_logs)))
    plt.subplot(4, 1, 3)
    plt.plot(mIoU_log)
    plt.title('max mIoU:{:f}'.format(np.max(mIoU_log)))
    plt.subplot(4, 1, 4)
    plt.axis('off')
    plt.text(0.1, 0.7, "MAE: {:f}".format(MAE))
    plt.text(0.1, 0.4, "AUC: {:f}".format(AUC))
    plt.text(0.1, 0.1, "AP: {:f}".format(AP))
    plt.savefig(out)
    plt.show()



if __name__ == '__main__':
    fire.Fire(metrics)
    exit(0)