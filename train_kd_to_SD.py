'''
Following distilaltion distills knowledge from a normal teacher to
a self distill student model. This enables us with the option of 
where (meaning at what FC layer) to transfer the KD loss, instead of
just one option of tranferring to the final FC layer of the original 
student model.
'''
# train a student network distilling from teacher

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam


from tqdm import tqdm
import argparse
import os
import logging
import numpy as np

from utils.utils import RunningAverage, set_logger, Params
from model import *
# this is for importing self_distillation resnet models
from resnet_self import *
from mobilenetv2_self import MobileNetV2_self
from data_loader import fetch_dataloader


# ************************** random seed **************************
seed = 100

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ************************** parameters **************************
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='experiments/S_resnet18_from_nasty/', type=str)
parser.add_argument('--teacher_resume', default=None, type=str,
                    help='If you specify the teacher resume here, we will use it instead of parameters from json file')
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--gpu_id', default=[0], type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

device_ids = args.gpu_id
torch.cuda.set_device(device_ids[0])


def loss_fn_kd(outputs, labels, teacher_outputs, epoch, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    """
    alpha = params.alpha
    T = params.temperature
    loc = params.loc
    if (epoch > params.kd_delay):
        KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs[loc]/T, dim=1),
                    F.softmax(teacher_outputs/T, dim=1))* (alpha * T * T) + \
              nn.CrossEntropyLoss()(outputs[loc], labels) * (1. - alpha)  +\
              (0.3*nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs[1]/T, dim=1),
                    F.softmax(outputs[0]/T, dim=1)) +  0.7*nn.CrossEntropyLoss()(outputs[1], labels)) + \
              nn.CrossEntropyLoss()(outputs[0], labels) 
    
    else:
        KD_loss = nn.CrossEntropyLoss()(outputs[0], labels)
    return KD_loss


# ************************** training function **************************
def train_epoch_kd(model, t_model, epoch, optim, loss_fn_kd, data_loader, params):
    model.train()
    t_model.eval()
    loss_avg = RunningAverage()

    with tqdm(total=len(data_loader)) as t:  # Use tqdm for progress bar
        for i, (train_batch, labels_batch) in enumerate(data_loader):
            if params.cuda:
                train_batch = train_batch.cuda()  # (B,3,32,32)
                labels_batch = labels_batch.cuda()  # (B,)

            # compute model output and loss
            output_batch_list, _ = model(train_batch)  # logit without SoftMax

            # get one batch output from teacher_outputs list
            with torch.no_grad():
                output_teacher_batch = t_model(train_batch)   # logit without SoftMax

            # CE(output, label) + KLdiv(output, teach_out)
            loss = loss_fn_kd(output_batch_list, labels_batch, output_teacher_batch, epoch, params)

            optim.zero_grad()
            loss.backward()
            optim.step()

            # update the average loss
            loss_avg.update(loss.item())

            # tqdm setting
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    return loss_avg()


def evaluate(model, loss_fn, data_loader, params, is_self=False):
    model.eval()
    # summary for current eval loop
    summ = []
    with torch.no_grad():
        # compute metrics over the dataset
        for data_batch, labels_batch in data_loader:
            # if params.cuda:
            data_batch = data_batch.cuda()          # (B,3,32,32)
            labels_batch = labels_batch.cuda()      # (B,)

            # compute model output
            if is_self:
                output_batch_list, _ = model(data_batch)
                output_batch = output_batch_list[0]
            else:
                output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.cpu().numpy()
            labels_batch = labels_batch.cpu().numpy()
            # calculate accuracy
            output_batch = np.argmax(output_batch, axis=1)
            acc = 100.0 * np.sum(output_batch == labels_batch) / float(labels_batch.shape[0])
            summary_batch = {'acc': acc, 'loss': loss.item()}
            summ.append(summary_batch)
    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    return metrics_mean

def evaluate_self(model, loss_fn, data_loader, params):
    model.eval()
    # summary for current eval loop
    with torch.no_grad():
        correct = [0 for _ in range(5)]
        predicted = [0 for _ in range(5)]
        total = 0.0
        for data in data_loader:
            model.eval()
            model.to('cuda')
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs, outputs_feature = model(images)
            ensemble = sum(outputs) / len(outputs)
            outputs.append(ensemble)
            for classifier_index in range(len(outputs)):
                _, predicted[classifier_index] = torch.max(outputs[classifier_index].data, 1)
                correct[classifier_index] += float(predicted[classifier_index].eq(labels.data).cpu().sum())
            total += float(labels.size(0))
        print('Test Set AccuracyAcc: 4/4: %.4f%% 3/4: %.4f%% 2/4: %.4f%%  1/4: %.4f%%'
                ' Ensemble: %.4f%%' % (100 * correct[0] / total, 100 * correct[1] / total,
                                        100 * correct[2] / total, 100 * correct[3] / total,
                                        100 * correct[4] / total))


def train_and_eval_kd(model, t_model, optim, loss_fn, train_loader, dev_loader, params):
    best_val_acc = -1
    best_epo = -1
    lr = params.learning_rate
    best_testAcc = [0.0]
    train_loss_list = [0.0]
    val_loss_list = [0.0]
    for epoch in range(params.num_epochs):
        # LR schedule *****************
        lr = adjust_learning_rate(optim, epoch, lr, params)
        
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        logging.info('Learning Rate {}'.format(lr))

        # ********************* one full pass over the training set *********************
        train_loss = train_epoch_kd(model, t_model, epoch, optim, loss_fn, train_loader, params)
        logging.info("- Train loss : {:05.3f}".format(train_loss))
        train_loss_list.append(train_loss)
        # ********************* Evaluate for one epoch on validation set *********************
        val_metrics = evaluate(model, nn.CrossEntropyLoss(), dev_loader, params, is_self=True)  # {'acc':acc, 'loss':loss}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics.items())
        logging.info("- Eval metrics : " + metrics_string)
        val_acc = val_metrics['acc']
        
        if val_acc > max(best_testAcc):
                print("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n". format(val_acc))
                torch.save(model.state_dict(), args.save_path+"S-{}_T-{}_{}_Tmp_{}_alp_{}_KDLoc{}_ep_{}_testAcc_{}.pt".\
                    format(params.model_name, params.teacher_model, params.dataset, params.temperature, params.alpha,\
                        params.loc, params.num_epochs, val_acc))
                print("\n>_ Deleting previous model file with accuracy {:.3f}% now...\n".format(max(best_testAcc)))
                if len(best_testAcc) > 1:
                    os.remove(args.save_path+"S-{}_T-{}_{}_Tmp_{}_alp_{}_KDLoc{}_ep_{}_testAcc_{}.pt".\
                    format(params.model_name, params.teacher_model, params.dataset, params.temperature, params.alpha,\
                        params.loc, params.num_epochs, max(best_testAcc)))
        best_testAcc.append(val_acc)
        val_loss_list.append(val_metrics['loss'])

        # save model
        save_name = os.path.join(args.save_path, 'last_model.tar')
        torch.save({
            'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optim.state_dict()},
            save_name)

        # ********************* get the best validation accuracy *********************
      
        if val_acc >= best_val_acc:
            best_epo = epoch + 1
            best_val_acc = val_acc
            logging.info('- New best model ')
            # save best model
            save_name = os.path.join(args.save_path, 'best_model.tar')
            torch.save({
                'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optim.state_dict()},
                save_name)

        logging.info('- So far best epoch: {}, best acc: {:05.3f}'.format(best_epo, best_val_acc))

    # ********************* This part is for writing the training loss and test acc to a text file *******************
    file_name1 = os.path.join(args.save_path + 'S-{}_T-{}_{}_Tmp_{}_alp_{}_KDLoc{}_ep_{}_testAcc.txt')
    file_name2 = os.path.join(args.save_path + 'S-{}_T-{}_{}_Tmp_{}_alp_{}_KDLoc{}_ep_{}_trainLoss.txt')
    file_name3 = os.path.join(args.save_path + 'S-{}_T-{}_{}_Tmp_{}_alp_{}_KDLoc{}_ep_{}_testLoss.txt')
    with open(file_name1.\
        format(params.model_name, params.teacher_model, params.dataset, params.temperature, params.alpha,\
            params.loc, params.num_epochs), 'w') as f:
        for item in best_testAcc:
            f.write("%s\n" % item)
    with open(file_name2.\
        format(params.model_name, params.teacher_model, params.dataset, params.temperature, params.alpha,\
            params.loc, params.num_epochs), 'w') as f:
        for item in train_loss_list:
            f.write("%s\n" % item)
    with open(file_name3.\
        format(params.model_name, params.teacher_model, params.dataset, params.temperature, params.alpha,\
            params.loc, params.num_epochs), 'w') as f:
        for item in val_loss_list:
            f.write("%s\n" % item)


def adjust_learning_rate(opt, epoch, lr, params):
    if epoch in params.schedule:
        lr = lr * params.gamma
        for param_group in opt.param_groups:
            param_group['lr'] = lr
    return lr


if __name__ == "__main__":
    # ************************** set log **************************
    set_logger(os.path.join(args.save_path, 'training.log'))

    # #################### Load the parameters from json file #####################################
    json_path = os.path.join(args.save_path, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    params.cuda = torch.cuda.is_available() # use GPU if available

    for k, v in params.__dict__.items():
        logging.info('{}:{}'.format(k, v))

    # ########################################## Dataset ##########################################
    trainloader = fetch_dataloader('train', params)
    devloader = fetch_dataloader('dev', params)

    # ############################################ Model ############################################
    if params.dataset == 'cifar10':
        num_class = 10
    elif params.dataset == 'cifar100':
        num_class = 100
    else:
        num_class = 10

    logging.info('Number of class: ' + str(num_class))

    # ############################### Student Model here are self distill models###############################
    logging.info('Create Student Model --- ' + params.model_name)

    # ResNet 18 / 34 / 50 ****************************************
    if params.model_name == 'resnet18':
        model = resnet18(num_class=num_class)
    elif params.model_name == 'resnet50':
        model = resnet50(num_class=num_class)
    elif params.model_name == 'mobilenetv2':
        model = MobileNetV2_self(class_num=num_class)

    else:
        model = None
        print('Not support for model ' + str(params.model_name))
        exit()

    # ############################### Teacher Model ###############################
    logging.info('Create Teacher Model --- ' + params.teacher_model)
    # ResNet 18 / 34 / 50 ****************************************
    if params.teacher_model == 'resnet18':
        teacher_model = ResNet18(num_class=num_class)
    elif params.teacher_model == 'resnet34':
        teacher_model = ResNet34(num_class=num_class)
    elif params.teacher_model == 'resnet50':
        teacher_model = ResNet50(num_class=num_class)

    # PreResNet(ResNet for CIFAR-10)  20/32/56/110 ***************
    elif params.teacher_model.startswith('preresnet20'):
        teacher_model = PreResNet(depth=20)
    elif params.teacher_model.startswith('preresnet32'):
        teacher_model = PreResNet(depth=32)
    elif params.teacher_model.startswith('preresnet56'):
        teacher_model = PreResNet(depth=56)
    elif params.teacher_model.startswith('preresnet110'):
        teacher_model = PreResNet(depth=110)

    # DenseNet *********************************************
    elif params.teacher_model == 'densenet121':
        teacher_model = densenet121(num_class=num_class)
    elif params.teacher_model == 'densenet161':
        teacher_model = densenet161(num_class=num_class)
    elif params.teacher_model == 'densenet169':
        teacher_model = densenet169(num_class=num_class)

    # ResNeXt *********************************************
    elif params.teacher_model == 'resnext29':
        teacher_model = CifarResNeXt(cardinality=8, depth=29, num_classes=num_class)

    elif params.teacher_model == 'mobilenetv2':
        teacher_model = MobileNetV2(class_num=num_class)

    elif params.teacher_model == 'shufflenetv2':
        teacher_model = shufflenetv2(class_num=num_class)

    elif params.teacher_model == 'net':
        teacher_model = Net(num_class, args)

    elif params.teacher_model == 'mlp':
        teacher_model = MLP(num_class=num_class)

    else:
        teacher_model = None
        exit()

    if params.cuda:
        model = model.cuda()
        teacher_model = teacher_model.cuda()

    if len(args.gpu_id) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        teacher_model = nn.DataParallel(teacher_model, device_ids=device_ids)

    # checkpoint ********************************
    if args.resume:
        logging.info('- Load checkpoint model from {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        logging.info('- Train from scratch ')

    # load teacher model
    if args.teacher_resume:
        teacher_resume = args.teacher_resume
        logging.info('------ Teacher Resume from system parameters!')
    else:
        teacher_resume = params.teacher_resume
    logging.info('- Load Trained teacher model from {}'.format(teacher_resume))
    checkpoint = torch.load(teacher_resume)
    teacher_model.load_state_dict(checkpoint['state_dict'])

    # ############################### Optimizer ###############################
    if params.model_name == 'net' or params.model_name == 'mlp':
        optimizer = Adam(model.parameters(), lr=params.learning_rate)
        logging.info('Optimizer: Adam')
    else:
        optimizer = SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
        logging.info('Optimizer: SGD')

    # ************************** LOSS **************************
    criterion = loss_fn_kd

    # ************************** Teacher ACC **************************
    logging.info("- Teacher Model Evaluation ....")
    val_metrics = evaluate(teacher_model, nn.CrossEntropyLoss(), devloader, params)  # {'acc':acc, 'loss':loss}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics.items())
    logging.info("- Teacher Model Eval metrics : " + metrics_string)

    # ************************** train and evaluate **************************
    train_and_eval_kd(model, teacher_model, optimizer, criterion, trainloader, devloader, params)
    # #This is for self_dist_student eval, where we should see nonrandom acc at the final output,
    # # and the corresponding FC layer output where we used KD from the nasty teacher.
    evaluate_self(model, nn.CrossEntropyLoss(), devloader, params)
