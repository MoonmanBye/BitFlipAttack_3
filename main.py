import argparse
import time
import datetime
import datasets
import models
from utils import *
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, L1Loss
from torch.optim import SGD, Adam
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Improving stealthy BFA robustness via output code matching')
parser.add_argument('--data_dir', type=str, default='data/')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset for processing')
parser.add_argument('--num_classes', '-c', default=10, type=int, help='number of classes in the dataset')
parser.add_argument('--arch', '-a', type=str, default='resnet20_quan', help='model architecture')
parser.add_argument('--bits', type=int, default=8, help='quantization bits')
parser.add_argument('--ocm', action='store_true', help='output layer coding with bit strings')
parser.add_argument('--output_act', type=str, default='linear', help='output act. (only linear and tanh is supported)')
parser.add_argument('--code_length', '-cl', default=16, type=int, help='length of codewords')
parser.add_argument('--outdir', type=str, default='results/', help='folder to save model and training log')
parser.add_argument('--epochs', '-e', default=160, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch', '-b', default=128, type=int, metavar='N', help='Mini-batch size (default: 128)')
parser.add_argument('--opt', type=str, default='sgd', help='sgd or adam optimizer')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--schedule', type=str, default='step', help='learning rate schedule')
parser.add_argument('--weight-decay', '-wd', default=1e-4, type=float, help='weight decay (default: 1e-4 for OCM)')
parser.add_argument('--gpu', default="0", type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=250, type=int, help='print frequency (default: 250)')
parser.add_argument('--clustering', '-pc', action='store_true', help='add piecewise clustering regularization')
parser.add_argument('--lambda_coeff', '-lam', type=float, default=1e-3, help='piecewise clustering strength')
parser.add_argument('--eval', action="store_true", help='load saved model weights from outdir path to evaluate only')
parser.add_argument('--resume', action="store_true", help='resume training from outdir path')
parser.add_argument('--finetune', action="store_true", help='for finetuning pre-trained imagenet models')
parser.add_argument('--ft_path', type=str, default='results/imagenet/resnet50_quan8/', help='finetune model path')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

np.random.seed(args.seed)
torch.manual_seed(args.seed)
gpu_list = [int(i) for i in args.gpu.strip().split(",")] if args.gpu is not "0" else [0]
if args.gpu == "-1":
    device = torch.device('cpu')
    print('Using cpu')
else:
    device = torch.device('cuda')
    print('Using gpu: ' + args.gpu)




def train(loader, model, criterion, optimizer, epoch, C):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(loader), [batch_time, data_time, losses, top1, top5], prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    for i, data in enumerate(loader):
        data_time.update(time.time() - end)

        inputs, targets = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if args.clustering:
            loss += clustering_loss(model, args.lambda_coeff)

        if args.ocm:
            output_probs = lambda z: F.softmax(torch.log(F.relu(torch.matmul(z, C.T)) + 1e-6))
            probs = output_probs(outputs)
            labels = torch.tensor([torch.where(torch.all(C == targets[i], dim=1))[0][0] for i in range(targets.shape[0])])
            acc1, acc5 = accuracy(probs, labels.to(device), topk=(1, 5))
        else:
            acc1, acc5 = accuracy(nn.Softmax()(outputs), targets, topk=(1, 5))

        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        loss.backward()
        
        #add random noise for conv1
        ori_conv1_grad = model.module.conv1.weight.grad.clone()
        var_conv1_list.append(torch.var(ori_conv1_grad, unbiased=False))
        ori_conv1_grad = torch.autograd.Variable(ori_conv1_grad, requires_grad=True)         
        rand_conv1_grad = torch.rand_like(ori_conv1_grad).cuda()
        loss_conv1_grad = criterion_grad(ori_conv1_grad,rand_conv1_grad)
        loss_conv1_grad.backward()
        #Above are added codes
        
        #add random noise for layer1[0] conv1
        ori_layer101_grad = model.module.layer1[0].conv1.weight.grad.clone()
        var_layer101_list.append(torch.var(ori_layer101_grad, unbiased=False))
        ori_layer101_grad = torch.autograd.Variable(ori_layer101_grad, requires_grad=True)         
        rand_layer101_grad = torch.rand_like(ori_layer101_grad).cuda()
        loss_layer101_grad = criterion_grad(ori_layer101_grad,rand_layer101_grad)
        loss_layer101_grad.backward()
        #Above are added codes      
 
        #add random noise for layer1[0] conv2
        ori_layer102_grad = model.module.layer1[0].conv2.weight.grad.clone()
        var_layer102_list.append(torch.var(ori_layer102_grad, unbiased=False))
        ori_layer102_grad = torch.autograd.Variable(ori_layer102_grad, requires_grad=True)         
        rand_layer102_grad = torch.rand_like(ori_layer102_grad).cuda()
        loss_layer102_grad = criterion_grad(ori_layer102_grad,rand_layer102_grad)
        loss_layer102_grad.backward()
        #Above are added codes 
        
        #add random noise for layer1[1] conv1
        ori_layer111_grad = model.module.layer1[1].conv1.weight.grad.clone()
        var_layer111_list.append(torch.var(ori_layer111_grad, unbiased=False))
        ori_layer111_grad = torch.autograd.Variable(ori_layer111_grad, requires_grad=True)         
        rand_layer111_grad = torch.rand_like(ori_layer111_grad).cuda()
        loss_layer111_grad = criterion_grad(ori_layer111_grad,rand_layer111_grad)
        loss_layer111_grad.backward()
        #Above are added codes 
        
        #add random noise for layer1[1] conv2
        ori_layer112_grad = model.module.layer1[1].conv2.weight.grad.clone()
        var_layer112_list.append(torch.var(ori_layer112_grad, unbiased=False))
        ori_layer112_grad = torch.autograd.Variable(ori_layer112_grad, requires_grad=True)         
        rand_layer112_grad = torch.rand_like(ori_layer112_grad).cuda()
        loss_layer112_grad = criterion_grad(ori_layer112_grad,rand_layer112_grad)
        loss_layer112_grad.backward()
        #Above are added codes 
        
        #add random noise for layer1[2] conv1
        ori_layer121_grad = model.module.layer1[2].conv1.weight.grad.clone()
        var_layer121_list.append(torch.var(ori_layer121_grad, unbiased=False))
        ori_layer121_grad = torch.autograd.Variable(ori_layer121_grad, requires_grad=True)         
        rand_layer121_grad = torch.rand_like(ori_layer121_grad).cuda()
        loss_layer121_grad = criterion_grad(ori_layer121_grad,rand_layer121_grad)
        loss_layer121_grad.backward()
        #Above are added codes 
        
        #add random noise for layer1[2] conv2
        ori_layer122_grad = model.module.layer1[2].conv2.weight.grad.clone()
        var_layer122_list.append(torch.var(ori_layer122_grad, unbiased=False))
        ori_layer122_grad = torch.autograd.Variable(ori_layer122_grad, requires_grad=True)         
        rand_layer122_grad = torch.rand_like(ori_layer122_grad).cuda()
        loss_layer122_grad = criterion_grad(ori_layer122_grad,rand_layer122_grad)
        loss_layer122_grad.backward()
        #Above are added codes 
        
        #add random noise for layer2[0] conv1
        ori_layer201_grad = model.module.layer2[0].conv1.weight.grad.clone()
        var_layer201_list.append(torch.var(ori_layer201_grad, unbiased=False))
        ori_layer201_grad = torch.autograd.Variable(ori_layer201_grad, requires_grad=True)         
        rand_layer201_grad = torch.rand_like(ori_layer201_grad).cuda()
        loss_layer201_grad = criterion_grad(ori_layer201_grad,rand_layer201_grad)
        loss_layer201_grad.backward()
        #Above are added codes 
        
        #add random noise for layer2[0] conv2
        ori_layer202_grad = model.module.layer2[0].conv2.weight.grad.clone()
        var_layer202_list.append(torch.var(ori_layer202_grad, unbiased=False))
        ori_layer202_grad = torch.autograd.Variable(ori_layer202_grad, requires_grad=True)         
        rand_layer202_grad = torch.rand_like(ori_layer202_grad).cuda()
        loss_layer202_grad = criterion_grad(ori_layer202_grad,rand_layer202_grad)
        loss_layer202_grad.backward()
        #Above are added codes 
        
        #add random noise for layer2[1] conv1
        ori_layer211_grad = model.module.layer2[1].conv1.weight.grad.clone()
        var_layer211_list.append(torch.var(ori_layer211_grad, unbiased=False))
        ori_layer211_grad = torch.autograd.Variable(ori_layer211_grad, requires_grad=True)         
        rand_layer211_grad = torch.rand_like(ori_layer211_grad).cuda()
        loss_layer211_grad = criterion_grad(ori_layer211_grad,rand_layer211_grad)
        loss_layer211_grad.backward()
        #Above are added codes
        
        #add random noise for layer2[1] conv2
        ori_layer212_grad = model.module.layer2[1].conv2.weight.grad.clone()
        var_layer212_list.append(torch.var(ori_layer212_grad, unbiased=False))
        ori_layer212_grad = torch.autograd.Variable(ori_layer212_grad, requires_grad=True)         
        rand_layer212_grad = torch.rand_like(ori_layer212_grad).cuda()
        loss_layer212_grad = criterion_grad(ori_layer212_grad,rand_layer212_grad)
        loss_layer212_grad.backward()
        #Above are added codes
        
        #add random noise for layer2[2] conv1
        ori_layer221_grad = model.module.layer2[2].conv1.weight.grad.clone()
        var_layer221_list.append(torch.var(ori_layer221_grad, unbiased=False))
        ori_layer221_grad = torch.autograd.Variable(ori_layer221_grad, requires_grad=True)         
        rand_layer221_grad = torch.rand_like(ori_layer221_grad).cuda()
        loss_layer221_grad = criterion_grad(ori_layer221_grad,rand_layer221_grad)
        loss_layer221_grad.backward()
        #Above are added codes
        
        #add random noise for layer2[2] conv2
        ori_layer222_grad = model.module.layer2[2].conv2.weight.grad.clone()
        var_layer222_list.append(torch.var(ori_layer222_grad, unbiased=False))
        ori_layer222_grad = torch.autograd.Variable(ori_layer222_grad, requires_grad=True)         
        rand_layer222_grad = torch.rand_like(ori_layer222_grad).cuda()
        loss_layer222_grad = criterion_grad(ori_layer222_grad,rand_layer222_grad)
        loss_layer222_grad.backward()
        #Above are added codes
        
        #add random noise for layer3[0] conv1
        ori_layer301_grad = model.module.layer3[0].conv1.weight.grad.clone()
        var_layer301_list.append(torch.var(ori_layer301_grad, unbiased=False))
        ori_layer301_grad = torch.autograd.Variable(ori_layer301_grad, requires_grad=True)         
        rand_layer301_grad = torch.rand_like(ori_layer301_grad).cuda()
        loss_layer301_grad = criterion_grad(ori_layer301_grad,rand_layer301_grad)
        loss_layer301_grad.backward()
        #Above are added codes
        
        #add random noise for layer3[0] conv2
        ori_layer302_grad = model.module.layer3[0].conv2.weight.grad.clone()
        var_layer302_list.append(torch.var(ori_layer302_grad, unbiased=False))
        ori_layer302_grad = torch.autograd.Variable(ori_layer302_grad, requires_grad=True)         
        rand_layer302_grad = torch.rand_like(ori_layer302_grad).cuda()
        loss_layer302_grad = criterion_grad(ori_layer302_grad,rand_layer302_grad)
        loss_layer302_grad.backward()
        #Above are added codes
        
        #add random noise for layer3[1] conv1
        ori_layer311_grad = model.module.layer3[1].conv1.weight.grad.clone()
        var_layer311_list.append(torch.var(ori_layer311_grad, unbiased=False))
        ori_layer311_grad = torch.autograd.Variable(ori_layer311_grad, requires_grad=True)         
        rand_layer311_grad = torch.rand_like(ori_layer311_grad).cuda()
        loss_layer311_grad = criterion_grad(ori_layer311_grad,rand_layer311_grad)
        loss_layer311_grad.backward()
        #Above are added codes
        
        #add random noise for layer3[1] conv2
        ori_layer312_grad = model.module.layer3[1].conv2.weight.grad.clone()
        var_layer312_list.append(torch.var(ori_layer312_grad, unbiased=False))
        ori_layer312_grad = torch.autograd.Variable(ori_layer312_grad, requires_grad=True)         
        rand_layer312_grad = torch.rand_like(ori_layer312_grad).cuda()
        loss_layer312_grad = criterion_grad(ori_layer312_grad,rand_layer312_grad)
        loss_layer312_grad.backward()
        #Above are added codes
        
        #add random noise for layer3[2] conv1
        ori_layer321_grad = model.module.layer3[2].conv1.weight.grad.clone()
        var_layer321_list.append(torch.var(ori_layer321_grad, unbiased=False))
        ori_layer321_grad = torch.autograd.Variable(ori_layer321_grad, requires_grad=True)         
        rand_layer321_grad = torch.rand_like(ori_layer321_grad).cuda()
        loss_layer321_grad = criterion_grad(ori_layer321_grad,rand_layer321_grad)
        loss_layer321_grad.backward()
        #Above are added codes
        
        #add random noise for layer3[2] conv2
        ori_layer322_grad = model.module.layer3[2].conv2.weight.grad.clone()
        var_layer322_list.append(torch.var(ori_layer322_grad, unbiased=False))
        ori_layer322_grad = torch.autograd.Variable(ori_layer322_grad, requires_grad=True)         
        rand_layer322_grad = torch.rand_like(ori_layer322_grad).cuda()
        loss_layer322_grad = criterion_grad(ori_layer322_grad,rand_layer322_grad)
        loss_layer322_grad.backward()
        #Above are added codes
        
        #add random noise for linear
        ori_linear_grad = model.module.linear.weight.grad.clone()
        var_linear_list.append(torch.var(ori_linear_grad, unbiased=False))
        ori_linear_grad = torch.autograd.Variable(ori_linear_grad, requires_grad=True)         
        rand_linear_grad = torch.rand_like(ori_linear_grad).cuda()
        loss_linear_grad = criterion_grad(ori_linear_grad,rand_linear_grad)
        loss_linear_grad.backward()
        #Above are added codes

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, top1.avg


def test(loader, model, criterion, C):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(loader), [batch_time, losses, top1, top5], prefix='Test: ')
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if args.clustering:
                loss += clustering_loss(model, args.lambda_coeff)

            if args.ocm:
                output_probs = lambda z: F.softmax(torch.log(F.relu(torch.matmul(z, C.T)) + 1e-6))
                probs = output_probs(outputs)
                labels = torch.tensor([torch.where(torch.all(C == targets[i], dim=1))[0][0] for i in range(targets.shape[0])])
                acc1, acc5 = accuracy(probs, labels.to(device), topk=(1, 5))
            else:
                acc1, acc5 = accuracy(nn.Softmax()(outputs), targets, topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return losses.avg, top1.avg


def main():
    # Load dataset and model architecture
    DATASET = datasets.__dict__[args.dataset](args)
    train_loader, test_loader = DATASET.loaders()

    
    #Basic function
    if args.ocm:
        n_output = args.code_length
        criterion = L1Loss()
        C = torch.tensor(DATASET.C).to(device)
    else:
        assert args.output_act == 'linear'
        n_output = args.num_classes
        criterion = CrossEntropyLoss()
        C = torch.tensor(np.eye(args.num_classes)).to(device)
    model = models.__dict__[args.arch](n_output, args.bits, args.output_act)
    model = nn.DataParallel(model, gpu_list).to(device) if len(gpu_list) > 1 else nn.DataParallel(model).to(device)
    
    if args.opt == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    log_filename = os.path.join(args.outdir, 'log.txt')
    
    if not args.eval:
        if args.resume:
            resume_best = torch.load(args.outdir + 'model_best.pth.tar', map_location=device)
            model.load_state_dict(resume_best['state_dict'])
            _, best_acc1 = test(test_loader, model, criterion, C)
            resume = torch.load(args.outdir + 'checkpoint.pth.tar', map_location=device)
            model.load_state_dict(resume['state_dict'])
            optimizer.load_state_dict(resume['optimizer'])
            start_epoch = resume['epoch']
        else:
            if args.finetune:
                pre_dict = torch.load(args.ft_path + 'model_best.pth.tar', map_location=device)['state_dict']
                pre_dict = {k: v for k, v in pre_dict.items() if 'module.linear' not in k}
                model.load_state_dict(pre_dict, strict=False)
                init_logfile(log_filename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")
                start_epoch, best_acc1 = 0, 0
            else:
                init_logfile(log_filename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")
                start_epoch, best_acc1 = 0, 0

        for epoch in range(start_epoch, args.epochs):
            lr = lr_scheduler(optimizer, epoch, args)

            before = time.time()
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, C)
            test_loss, test_acc = test(test_loader, model, criterion, C)
            after = time.time()

            is_best = test_acc > best_acc1
            best_acc1 = max(test_acc, best_acc1)

            save_checkpoint({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(),
                             'best_acc1': best_acc1, 'optimizer': optimizer.state_dict()}, is_best, args.outdir)

            log(log_filename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch, str(datetime.timedelta(seconds=(after - before))), lr, train_loss, train_acc, test_loss, test_acc))
    else:
        eval_best = torch.load(args.outdir + 'model_best.pth.tar', map_location=device)
        model.load_state_dict(eval_best['state_dict'])
        test(test_loader, model, criterion, C)
    



if __name__ == "__main__":
       
    #Some parameters
    var_conv1_list=[]
    var_layer101_list=[]
    var_layer102_list=[]
    var_layer111_list=[]
    var_layer112_list=[]
    var_layer121_list=[]
    var_layer122_list=[]
    var_layer201_list=[]
    var_layer202_list=[]
    var_layer211_list=[]
    var_layer212_list=[]
    var_layer221_list=[]
    var_layer222_list=[]
    var_layer301_list=[]
    var_layer302_list=[]
    var_layer311_list=[]
    var_layer312_list=[]
    var_layer321_list=[]
    var_layer322_list=[]
    var_linear_list=[]
    
    #needed function
    criterion_grad = nn.MSELoss()
    
    main()
    
    #added for gradient variance check
    for i in range(len(var_conv1_list)):
    var_conv1_list[i] = var_conv1_list[i].cpu().data
    fig = plt.figure(figsize=(16,8))
    plt.plot(var_conv1_list)
    plt.title('gradient variance for conv1 layer')
