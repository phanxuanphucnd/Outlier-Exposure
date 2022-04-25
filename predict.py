import argparse

import torch
from model import ClfGRU
from utils.utils import text_to_torch_tensor, prepare_data, predict

#  Prepare vocab
train_path = './data/v1.0/train.csv'
TEXT = prepare_data(train_path)

#  MODEL
#  Define model architecture

model = ClfGRU(
    num_classes=6,
    TEXT=TEXT
).cuda()

#  Load model fine_tune
model.load_state_dict(torch.load('./models/v1.0/OE/model_finetune.dict'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text", type=str,
        default="Thanks c nhìu nha",
        help="Text input to test model"
    )

    args = parser.parse_args()
    text_tensor = text_to_torch_tensor(args.text)

    print("Predict: \n")
    print(predict(model, text_tensor))
