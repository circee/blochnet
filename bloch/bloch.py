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
    # psi_dx  = dfx(x,psi)
    # psi_ddx = dfx(x,psi_dx)
    
    psi_real = psi[:,0].reshape(-1,1);psi_i = psi[:,1].reshape(-1,1)

    psi_real_dx = dfx(x,psi_real)
    psi_real_ddx = dfx(x,psi_real_dx)

    psi_i_dx = dfx(x,psi_i)
    psi_i_ddx = dfx(x,psi_i_dx)

    K = k**2
    
    #freal = (-1/2)*psi_real_ddx + ((1/2)*K+V-E)*psi_real - 2*k*psi_i_dx
    
    #fimag = (-1/2)*psi_i_ddx + ((1/2)*K+V-E)*psi_i + 2*k*psi_real_dx
    
    freal = (-1/2)*psi_real_ddx + ((1/2)*K+V-E)*psi_real + k*psi_i_dx #cross term 1/2
    
    fimag = (-1/2)*psi_i_ddx + ((1/2)*K+V-E)*psi_i - k*psi_real_dx #cross term 1/2
    
    Lreal = (freal.pow(2)).mean()
    
    Limag = (fimag.pow(2)).mean()

    
    return Lreal, Limag

def normLoss(nn, pts, period, k1, k2, epoch):
    x = torch.linspace(0,period,pts).to(cuda).reshape(-1,1)
    x.requires_grad = True
    k = ((k2-k1)*torch.rand(1)+k1).to(cuda).expand(pts).reshape(-1,1)
    k.requires_grad = True
    u,energy, NFE, ADJ = nn(x,k)

    ur = u[:,0].reshape(-1,1); ui = u[:,1].reshape(-1,1)

    #psi = torch.cos(k*x)*ur; psi_i = torch.sin(k*x)*ui

#     prob = psi.pow(2)+psi_i.pow(2)

#     psisum = torch.trapezoid(prob, x, dim=0)

#     normerror = 1-torch.sqrt(psisum)

    #prob = psi.pow(2) + psi_i.pow(2)
    prob = ur.pow(2) + ui.pow(2)
    

    probsum = torch.trapezoid(prob, x, dim=0)
    

    #normerror = 1-torch.sqrt(psisum)
    #normerror = (1/probsum.mean())-1
    p = probsum.mean()
    
    normerror = abs(torch.sqrt(p)-1)#.pow(2)
    return normerror
    
    #return (normerror.pow(2))#.mean()
    
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

    NN = NN[:,0].reshape(-1,1)
    
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

def bcImLoss(nn, pts, b1, b2, k1, k2, verbose = False):
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

    NN = NN[:,1].reshape(-1,1)
    
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
        self.actFsig = torch.nn.Sigmoid()
        self.actFsin = sin()
        self.actFlin = linear()
        self.actFtanh = tanh()
        
        self.ENactF = sin()
        self.FNactF = sin()
        
        eneurons = neurons #
        uneurons = 32
        
        self.Lin  = torch.nn.Linear(2, uneurons)
        self.L1  = torch.nn.Linear(uneurons, uneurons)
        self.L2  = torch.nn.Linear(uneurons, uneurons)
        self.L3  = torch.nn.Linear(uneurons, uneurons)
        self.L4  = torch.nn.Linear(uneurons, uneurons)
        self.out    = torch.nn.Linear(uneurons, 2)
        
        self.Ein  = torch.nn.Linear(1, eneurons)
        self.E1  = torch.nn.Linear(eneurons, eneurons)
        self.E2  = torch.nn.Linear(eneurons, eneurons)
        self.E3  = torch.nn.Linear(eneurons, eneurons)
        self.Eout    = torch.nn.Linear(eneurons, 1)
        
    def forward(self,x,k):
        
        nfe = (1/2)*k**2
        
        Ein = self.Ein(k); Enin = self.Ein(-1*k)
        
        E1 = self.E1(Ein); h = self.ENactF(E1); En1 = self.E1(Enin); h_n = self.ENactF(En1)
        
        E2 = self.E2(h); h  = self.ENactF(E2); En2 = self.E2(h_n); h_n  = self.ENactF(En2)   
        
        E3 = self.E3(h); h = self.actFlin(E3); En3 = self.E3(h_n); h_n = self.actFlin(En3)
        
        E = self.Eout(h+h_n)
        E_k = E#+nfe
        
        Lin = self.Lin(torch.cat((x,E_k),1))
        
        L1 = self.L1(Lin); h = self.FNactF(L1)
        
        L2 = self.L2(h); h  = self.FNactF(L2)   
        
        L3 = self.L3(h); h = self.actFlin(L3)
        
        L4 = self.L4(h); h = self.actFlin(L4)
        
        U = self.out(h)

        #Ureal  = (U[:,0]).reshape(-1,1);Ui  = (U[:,0]).reshape(-1,1) # pass U out and fork later?
        
        #Ein = self.Ein(torch.cat((Ureal,k),1))


        extra = [nfe, E]

        return U, E_k, nfe, extra
        #return Ureal, Ui, E_k, extra
    
def blochModel(eqParams, netParams,PATH, verbose = False):
    
    b1 = eqParams[0]; b2 = eqParams[1]; k1 = eqParams[2]; k2 = eqParams[3]; v = eqParams[4]
    
    print(eqParams)
    
    neurons = netParams[0]; points = netParams[1]; epochs = netParams[2]; lr = netParams[3]; minLoss = netParams[4]; lrdecay = netParams[5]
    
    nn1 = evpNet(neurons)
    nn1.to(cuda)
    nn2 = copy.deepcopy(nn1)

    # optimizer
    betas = [0.999, 0.9999] 
    optimizer = optim.Adam(nn1.parameters(), lr=lr, betas=betas)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=lrdecay, total_iters=epochs)
    
    loss = [] ; lossHistory = [] ; tHistory = [] ; EHistory = [] ; blochLossHistory = [] ; trivLossHistory = [] ; BCfLossHistory = [] ; BCdfLossHistory = [] ; trivfLossHistory = [] ; trivdfLossHistory = [] ; Residuals = [] ; nlossHistory = [] ; lrHistory = [] ; blochLossImaginaryHistory = [] ; BCimfLossHistory = [] ; BCimdfLossHistory = []
    tepochs = 0
    lossLim = 10e10 
    
    width = abs(b2+b1)
    
    #grid = (width*(torch.rand(points)).reshape(-1,1)).to(cuda).reshape(-1,1)
    #grid.sort() #?
    g = torch.linspace(0,width,points).to(cuda)
    #g.requires_grad = True
    
    grid = torch.cat((g,g))
    for i in range(points-2):
        grid = torch.cat((grid,g))
    grid = grid.reshape(-1,1)
    grid.requires_grad = True
    print(grid.size())
    
    kvec = torch.linspace(k1,k2,points).to(cuda)
    #kvec.requires_grad = True
    
    k = torch.column_stack((kvec,kvec))
    for i in range(points-2):
        k = torch.column_stack((k,kvec))
    k = torch.flatten(k)
    k = k.reshape(-1,1)
    k.requires_grad = True
    print(k.size())
    
    
    time_start = time.time()
    for epoch in range(epochs):
        
        loss = 0.0 ; BClossf = 0.0 ; blochloss = 0.0 ; bcfLoss = 0.0 ; BClossdf = 0.0 ; trivf = 0.0 ; trivdf = 0.0 ; nloss = 0.0 ;  blochlossi = 0.0 ; BCimlossf = 0.0 ; BCimlossdf = 0.0
        
            
        if epoch % 1000 == 0 and verbose == True:
            print("EPOCH:")
            print(epoch)
        
            
        resample = 1
        
        if epoch % resample  == 0 and epoch != 0:
            kvec = (((k2-k1)*torch.rand((points)))+k1).to(cuda)#.reshape(-1,1)

            k = torch.column_stack((kvec,kvec))
            for i in range(points-2):
                k = torch.column_stack((k,kvec))
            k = torch.flatten(k)
            k = k.reshape(-1,1)
            k.requires_grad = True
            
            g = (width*torch.rand((points))).to(cuda)

            grid = torch.cat((g,g))
            for i in range(points-2):
                grid = torch.cat((grid,g))
            grid = grid.reshape(-1,1)
            grid.requires_grad = True
            
        
        V = kronigPenney(grid, v, width, b1)
    
        NN, EK, NFE, ADJ = nn1(grid,k)
        
        EHistory.append(EK[0].cpu().data.numpy())
        
        if epoch % 1000 == 0 and verbose == True:
            print("u(x) MEAN:")
            print(NN.mean())
          
        # CALCULATE LOSS TERMS
        bloch_loss_real, bloch_loss_imaginary = blochLoss(NN, grid, EK, V, k)
        
        bc_loss_f, bc_loss_df = bcLoss(nn1, 1000, 0, width, k1, k2) # network, point pairs, left bound, right bound, k1, k2
        bc_im_loss_f, bc_im_loss_df = bcImLoss(nn1, 1000, 0, width, k1, k2)
        
        norm_loss = normLoss(nn1, 1000, width, k1, k2, epoch)
        
        # OPTIM 
        lossTotal = bloch_loss_real + bloch_loss_imaginary + bc_loss_f + bc_loss_df + bc_im_loss_f + bc_im_loss_df + norm_loss
        
        lossTotal.backward(retain_graph = False) 
        
        optimizer.step() # optimize 
        scheduler.step()
        
        loss += lossTotal.cpu().data.numpy() # sum total loss for this iteration
        blochloss += bloch_loss_real.cpu().data.numpy() # sum total loss for this iteration
        BClossf += bc_loss_f.cpu().data.numpy()
        BClossdf += bc_loss_df.cpu().data.numpy()
        trivf += norm_loss.cpu().data.numpy()
        blochlossi += bloch_loss_imaginary.cpu().data.numpy() # sum total loss for this iteration
        BCimlossf += bc_im_loss_f.cpu().data.numpy()
        BCimlossdf += bc_im_loss_df.cpu().data.numpy()
        
        optimizer.zero_grad() # nullify loss gradients
    
        lossHistory.append(loss) # append total loss history
        blochLossHistory.append(blochloss)
        BCfLossHistory.append(BClossf)
        BCdfLossHistory.append(BClossdf)
        trivfLossHistory.append(trivf)
        blochLossImaginaryHistory.append(blochlossi)
        BCimfLossHistory.append(BCimlossf)
        BCimdfLossHistory.append(BCimlossdf)
        
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
        
    isthisloss = [lossHistory, blochLossHistory, BCfLossHistory, BCdfLossHistory, trivfLossHistory, trivdfLossHistory, lrHistory, blochLossImaginaryHistory, BCimfLossHistory, BCimdfLossHistory]
        
    return nn2, nn1, runTime, tepochs, isthisloss, tHistory, EHistory

############################################################################################################################
