import logging

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm


logger = logging.getLogger("zebel-scanner")


def dice_coef(y_true, y_pred, smooth=1000.0):
    y_true_f = y_true.contiguous()
    y_pred_f = y_pred.contiguous()
    intersection = (y_true_f * y_pred_f).sum()
    return (2.0 * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


class EvaluatorFactory:
    """
    This class is used to get different versions of evaluators
    """

    def __init__(self):
        pass

    @staticmethod
    def get_evaluator(testType="rmse", cuda=True):
        if testType == "rmse":
            return DocumentMseEvaluator(cuda)


class DocumentMseEvaluator:
    """
    Evaluator class for softmax classification
    """

    def __init__(self, cuda):
        self.cuda = cuda

    def evaluate(self, model, iterator):
        model.eval()
        lossAvg = None
        with torch.no_grad():
            for img, target in tqdm(iterator):
                if self.cuda:
                    img, target = img.cuda(), target.cuda()

                response = model(Variable(img))
                # print (response[0])
                # print (target[0])
                loss = F.mse_loss(response, Variable(target.float()))
                loss = torch.sqrt(loss)
                if lossAvg is None:
                    lossAvg = loss
                else:
                    lossAvg += loss
                # logger.debug("Cur loss %s", str(loss))

        lossAvg /= len(iterator)
        logger.info("Avg Val Loss %s", str((lossAvg).cpu().data.numpy()))


class Evaluator:
    """
    Evaluator class for softmax classification
    """

    def __init__(self, cuda):
        self.cuda = cuda
        self.criterion = dice_loss

    def evaluate(self, model, iterator):
        model.eval()
        lossAvg = None
        with torch.no_grad():
            for img, target in tqdm(iterator):
                if self.cuda:
                    img, target = img.cuda(), target.cuda()

                y_pred = model(img)
                loss = self.criterion(y_pred, Variable(target))
                if lossAvg is None:
                    lossAvg = loss
                else:
                    lossAvg += loss
                logger.debug("Cur loss %s", str(loss))

        lossAvg /= len(iterator)
        logger.info("Avg Val Loss %s", str((lossAvg).cpu().data.numpy()))


class GenericTrainer:
    """
    Base class for trainer; to implement a new training routine, inherit from this.
    """

    def __init__(self):
        pass


class Trainer:
    def __init__(self, train_iterator: DataLoader, model, cuda, optimizer):
        super().__init__()
        self.cuda = cuda
        self.train_iterator = train_iterator
        self.model = model
        self.optimizer = optimizer
        self.criterion = dice_loss

    def update_lr(self, epoch, schedule, gammas):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group["lr"]
                    param_group["lr"] = self.current_lr * gammas[temp]
                    logger.debug(
                        "Changing learning rate from %0.9f to %0.9f",
                        self.current_lr,
                        self.current_lr * gammas[temp],
                    )
                    self.current_lr *= gammas[temp]

    def train(self, epoch):
        self.model.train()
        lossAvg = None
        for img, target in tqdm(self.train_iterator):
            if self.cuda:
                img, target = img.cuda(), target.cuda()
            self.optimizer.zero_grad()
            # remove variable
            # print("img.shape", img.shape, img.dtype)
            y_pred = self.model(img)
            # print (response[0])
            # print (target[0])
            loss = self.criterion(Variable(target), y_pred)
            if lossAvg is None:
                lossAvg = loss
            else:
                lossAvg += loss
            # logger.debug("Cur loss %s", str(loss))
            loss.backward()
            self.optimizer.step()

        lossAvg /= len(self.train_iterator)
        logger.info("Avg Loss %s", str((lossAvg).cpu().data.numpy()))
