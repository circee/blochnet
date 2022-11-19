#################################################################################################
# Bloch network cross term maybe fix idk
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
#################################################################################################

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import grad
import matplotlib.pyplot as plot
import time
import copy
from os import path
import sys

cuda = torch.device('cuda')
print(torch.cuda.get_device_name(cuda))
print(torch.version.cuda)
print(torch.default_generator)

def dfx(x,f):
    return grad([f], [x], grad_outputs=torch.ones(x.shape,device=cuda), create_graph=True)[0]

# activation function options 
class tanh(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.tanh(input)
    

class sin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)
    
class linear(torch.nn.Module):
    @staticmethod
    def forward(input):
        return input
    
    
# bloch sde loss
def blochLoss(psi, x, E, V, k):
    psi_dx  = dfx(x,psi)
    psi_ddx = dfx(x,psi_dx)
    
    K = k**2
    
    f = ((K/2)+V-E)*psi - psi_ddx/2 
    
    L = (f.pow(2)).mean(); 
    
    return L

def normLoss(nn, pts, period, k1, k2, epoch):
    x = torch.linspace(0,period,pts).to(cuda).reshape(-1,1)
    x.requires_grad = True
    k = ((k2-k1)*torch.rand(1)+k1).to(cuda).expand(pts).reshape(-1,1)
    k.requires_grad = True
    u,energy, NFE, ADJ = nn(x,k)

    psi = torch.cos(k*x)*u

    prob = psi.pow(2)

    psisum = torch.trapezoid(prob, x, dim=0)

    normerror = 1-torch.sqrt(psisum)
    
    return (normerror.pow(2)).mean()
    
def bcLoss(nn, pts, b1, b2, k1, k2, verbose = False):
    # nn IN THIS CASE IS THE NETWORK NN1 NOT EVALUATED MODEL
    # pts is the number of DISTINCT PAIRS
    
    # set up tensors
    kvec = ((k2-k1)*torch.rand(pts)+k1).to(cuda).reshape(-1,1)
    kvec.requires_grad = True
    
    b1 = torch.linspace(b1,b1,pts).to(cuda).reshape(-1,1)
    b1.requires_grad = True
    
    b2 = torch.linspace(b2,b2,pts).to(cuda).reshape(-1,1)
    b2.requires_grad = True
    
    grid = torch.cat((b1,b2)).to(cuda).reshape(-1,1)
    
    kvec = torch.cat((kvec,kvec))

    # forward pass
    
    NN,KN, NFE, ADJ = nn(grid,kvec)
    
    DNN = dfx(grid,NN)
    
    NN = torch.chunk(NN,2)

    bv1 = NN[0];bv2 = NN[1]
    
    DNN = torch.chunk(DNN,2)
    
    bv1 = NN[0];bv2 = NN[1]

    ferr = bv1-bv2
    
    # ENFORCE DERIVATIVE BOUNDARY MATCHING
    
    dbv1 = DNN[0];dbv2 = DNN[1]
    
    dferr = dbv1-dbv2
    
    L = (ferr.pow(2)).mean()
    
    DL = (dferr.pow(2)).mean()
    
    if verbose:
        print(grid)
        print("")
        print(kvec)
        print("")
        print(NN)
        print("")
        print(bv1)
        print("")
        print(bv2)
        print("")
        print(L)
        print("")
        print("-----------------------------")

    return L, DL

def kronigPenney(X, V, L, a):
    # Gives the potential V at each point
    X = X.cpu()
    X = X.data.numpy()

    # approximate square wave 
    Vnp = V/(1+((X-(0.5))/(a/2))**30)
    
    Vtorch = torch.from_numpy(Vnp)
    Vtorch = Vtorch.cuda()
    return Vtorch

class evpNet(torch.nn.Module):
    def __init__(self,neurons=100):
        super(evpNet,self).__init__()
        
        # Activation Function
        self.actFsin = sin()
        self.actFlin = linear()
        self.actFtanh = tanh()
        
        eneurons = neurons #
        uneurons = neurons
        
        self.Lin  = torch.nn.Linear(2, uneurons)
        self.L1  = torch.nn.Linear(uneurons, uneurons)
        self.L2  = torch.nn.Linear(uneurons, uneurons)
        self.L3  = torch.nn.Linear(uneurons, uneurons)
        self.out    = torch.nn.Linear(uneurons, 1)
        
        self.Ein  = torch.nn.Linear(1, eneurons)
        self.E1  = torch.nn.Linear(eneurons, eneurons)
        self.E2  = torch.nn.Linear(eneurons, eneurons)
        self.E3  = torch.nn.Linear(eneurons, eneurons)
        self.Eout    = torch.nn.Linear(uneurons, 1)
        
    def forward(self,x,k):
        
        nfe = (1/2)*k**2
        
        Lin = self.Lin(torch.cat((x,k),1))
        
        L1 = self.L1(Lin)
        h1 = self.actFtanh(L1)
        
        L2 = self.L2(h1)
        h2  = self.actFtanh(L2)   
        
        L3 = self.L3(h2)
        h3 = self.actFlin(L3)
        
        U = self.out(h3)
        
        #Ein = self.Ein(torch.cat((U,k),1))
        Ein = self.Ein(k)
        
        E1 = self.E1(Ein)
        h1 = self.actFtanh(E1)
        
        E2 = self.E2(h1)
        h2  = self.actFtanh(E2)   
        
        E3 = self.E3(h2)
        h3 = self.actFlin(E3)
        
        E = self.Eout(h3)
        E_k = E

        return U, E_k, nfe, E
    
def blochModel(eqParams, netParams,PATH, verbose = False):
    
    b1 = eqParams[0]
    b2 = eqParams[1]
    k1 = eqParams[2]
    k2 = eqParams[3]
    v = eqParams[4]
    print(eqParams)
    neurons = netParams[0]
    points = netParams[1]
    epochs = netParams[2]
    lr = netParams[3]
    minLoss = netParams[4]
    lrdecay = netParams[5]
    
    nn1 = evpNet(neurons)
    nn1.to(cuda)
    nn2 = copy.deepcopy(nn1)

    # optimizer
    betas = [0.999, 0.9999] 
    optimizer = optim.Adam(nn1.parameters(), lr=lr, betas=betas)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=lrdecay, total_iters=epochs)
    
    loss = [] ; lossHistory = [] ; tHistory = [] ; EHistory = [] ; blochLossHistory = [] ; trivLossHistory = [] ; BCfLossHistory = [] ; BCdfLossHistory = [] ; trivfLossHistory = [] ; trivdfLossHistory = [] ; Residuals = [] ; nlossHistory = [] ; lrHistory = []
    tepochs = 0
    lossLim = 10e10 
    
    width = abs(b2+b1)
    
    grid = (width*(torch.rand(points)).reshape(-1,1)).to(cuda).reshape(-1,1)
    grid.sort()
    grid.requires_grad = True
    
    k = torch.linspace(k1,k2,points).to(cuda).reshape(-1,1)
    k.requires_grad = True
    
    time_start = time.time()
    for epoch in range(epochs):
        
        a = 1 ; b = 1
        
        loss = 0.0 ; BClossf = 0.0 ; blochloss = 0.0 ; BClossdf = 0.0 ; trivf = 0.0 ; trivdf = 0.0 ; nloss = 0.0 ; bcfLoss = 0.0
        
            
        if epoch % 1000 == 0 and verbose == True:
            print("EPOCH:")
            print(epoch)
        
            
        resample = 1
        if epoch % resample  == 0 and epoch != 0:
            k = (((k2-k1)*torch.rand((points)))+k1).to(cuda).reshape(-1,1)
            k.requires_grad = True
        
        V = kronigPenney(grid, v, width, b1)
    
        NN, EK, NFE, ADJ = nn1(grid,k)
        
        EHistory.append(EK[0].cpu().data.numpy())
        
        if epoch % 1000 == 0 and verbose == True:
            print("u(x) MEAN:")
            print(NN.mean())
          
        # CALCULATE LOSS TERMS
        bloch_loss = blochLoss(NN, grid, EK, V, k)
        
        bc_loss_f, bc_loss_df = bcLoss(nn1, 2000, 0, width, k1, k2) # network, point pairs, left bound, right bound, k1, k2
        
        norm_loss = normLoss(nn1, 2000, width, k1, k2, epoch)
        
        # OPTIM 
        lossTotal = bloch_loss + bc_loss_f + bc_loss_df + norm_loss
        
        lossTotal.backward(retain_graph = False) 
        
        optimizer.step() # optimize 
        scheduler.step()
        
        loss += lossTotal.cpu().data.numpy() # sum total loss for this iteration
        blochloss += bloch_loss.cpu().data.numpy() # sum total loss for this iteration
        BClossf += bc_loss_f.cpu().data.numpy()
        BClossdf += bc_loss_df.cpu().data.numpy()
        trivf += norm_loss.cpu().data.numpy()
        #trivdf += 0 #dftriv.data.numpy()
        
        optimizer.zero_grad() # nullify loss gradients
    
        lossHistory.append(loss) # append total loss history
        blochLossHistory.append(blochloss)
        BCfLossHistory.append(BClossf)
        BCdfLossHistory.append(BClossdf)
        trivfLossHistory.append(trivf)
        
        lrHistory.append(optimizer.param_groups[0]['lr'])
    
    
        if epoch % 1000 == 0 and verbose == True:
            print("LOSS:")
            print(lossHistory[-1])
            print("")
            print("NN2 LOSS:")
            print(lossLim)
            print("")
            print("RUNTIME:")
            print(str(round(time.time() - time_start,4)) + ' seconds')
            print("")
            print("LEARNING RATE:")
            print(optimizer.param_groups[0]['lr'])
            print("-----------------------------")
            
            
        # keep the best model (lowest loss) by using a deep copy
        if  lossTotal < lossLim:
            nn2 =  copy.deepcopy(nn1)
            lossLim=lossTotal

        # terminate training after loss threshold is reached 
        if lossTotal < minLoss and minLoss != -1 and epoch > 0.05*epochs:
            nn2 =  copy.deepcopy(nn1)
            print('Reached target loss')
            tepochs = epoch
            break
            
    time_end = time.time()
    runTime = time_end - time_start
    
    # print total runtime 
    print('Total runtime: ' + str(round(runTime,2)) + ' seconds')
    
    torch.save({
    'epoch': epoch,
    'model_state_dict': nn2.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': lossTotal,
    }, PATH)

    if tepochs == 0:
        tepochs = epochs
        
    isthisloss = [lossHistory, blochLossHistory, BCfLossHistory, BCdfLossHistory, trivfLossHistory, trivdfLossHistory, lrHistory]
        
    return nn2, nn1, runTime, tepochs, isthisloss, tHistory, EHistory

############################################################################################################################
