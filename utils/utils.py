import torch
from torch import LongTensor
import torch.nn.functional as F
from torchtext import data
from sklearn.preprocessing import binarize

TEXT = data.Field(
    pad_first=True,
    lower=True,
    fix_length=100
)


def prepare_data(path_data):

    """
    Builing format to store dataset. Build vocab of dataset.
    :param:
        path_data: path to train data : str
        fix_length: max length of sentence
    :return:
        TEXT: Dataset
    """

    data_wrapper = data.TabularDataset(
        path=path_data,
        format='csv',
        fields=[('text', TEXT)],
        skip_header=True
    )

    TEXT.build_vocab(data_wrapper, max_size=10000)
    return TEXT


def text_to_torch_tensor(text):

    """
    Convert Text input to Torch Tensor
    :param:
        text : str
    :return:
        torch tensor
    """

    # Tokenize text input
    test_sen = TEXT.preprocess(text)

    # Map list token in vocab
    test_sen = [[TEXT.vocab.stoi[x] for x in test_sen]]

    # Convert list to torch tensor 2D
    text_tensor = LongTensor(test_sen)
    return text_tensor.cuda()


def predict(model, text):
    logits = model(text)
    smax = F.softmax(logits - torch.max(logits, dim=1, keepdim=True)[0], dim=1)
    msp = -1 * torch.max(smax, dim=1)[0]
    temp_outlier_scores = msp.data.cpu().numpy().reshape(1, -1)
    return binarize(temp_outlier_scores, -0.5)[0]
