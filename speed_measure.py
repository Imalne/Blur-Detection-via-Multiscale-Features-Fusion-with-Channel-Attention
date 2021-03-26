from utils import getModel
from option import getPredictParser
import torch
import time

if __name__ == "__main__":
    parser = getPredictParser()
    args = parser.parse_args()
    net, _ = getModel(args, save_path=args.weight_path, mode="eval")
    input = torch.rand((1,3,320,320)).cuda()
    start = time.time()
    for i in range(1000):
        output = net(input)
    end = time.time()
    print((end-start)/1000)