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
from torch.distributions.beta import Beta
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
    
class normSin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin((355/113)*input)
    
class linear(torch.nn.Module):
    @staticmethod
    def forward(input):
        return input
    
class snek(torch.nn.Module):
    @staticmethod
    def forward(input):
        return (input + (torch.sin(input).pow(2)))
    
    
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

  # --------------------------------------------------------------------------------------  
    probreal = ur.pow(2) 
    probim = ui.pow(2)
    

    #probsumre = torch.trapezoid(probreal, x, dim=0)
    
    #probsumim = torch.trapezoid(probim, x, dim=0)
    
    ps = torch.trapezoid((probreal+probim), x, dim=0)
  # --------------------------------------------------------------------------------------  
    
    #orthosum = torch.trapezoid((ur+ui).pow(2), x, dim=0)
    

    #normerror = 1-torch.sqrt(psisum)
    #normerror = (1/probsum.mean())-1
    #p = probsum.mean()
    #o = orthosum.mean()
    
    #normerror = (torch.sqrt(p)-1).pow(2)
    #normerror = o + (torch.sqrt(p)-1).pow(2)
    
    #normerror = (1 - torch.sqrt(ps)).pow(2)
    normerror = (torch.sqrt(ps)-1).pow(2)
    
    return normerror.mean()
    
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

def kronigPenney(X, V, L, a, N):
    # Gives the potential V at each point
    #X = X.cpu()
    #X = X.data.numpy()

    # approximate square wave 
    Vnp = V/(1+((X-(0.5))/(a/2))**4)
    #Vnp = (V/(1+((X-(0.75))/(a/2))**4))+(-0.5*V/(1+((X-(0.25))/(a/2))**4))
    
    #Vtorch = torch.from_numpy(Vnp)
    #Vtorch = Vtorch.cuda()
    return Vnp#Vtorch

def coulombPotential(X, V, L, a, N):
    # Gives the potential V at each point
    #X = X.cpu()
    #X = X.data.numpy()

    Vnp = -V/(0.2+(10*(X-0.5))**2)
    
    #Vtorch = torch.from_numpy(Vnp)
    #Vtorch = Vtorch.cuda()
    return Vnp#Vtorch

def freeParticle(X, V, L, a, N):
    Vtorch = torch.zeros_like(X)
    return Vtorch

class evpNet(torch.nn.Module):
    def __init__(self,eneurons=64,uneurons=128):
        super(evpNet,self).__init__()
        
        # Activation Function
        self.actFsig = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.actFsin = sin()
        self.actFlin = linear()
        self.actFtanh = tanh()
        self.normSine = normSin()
        self.snek = snek()
        
        self.ENactF = snek()
        self.FNactF = tanh()
        
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
        self.E4  = torch.nn.Linear(eneurons, eneurons)
        self.Eout    = torch.nn.Linear(eneurons, 1)
        
    def forward(self,x,k):
        
        nfe = (1/2)*k**2
        
        Ein = self.Ein(k)#; Enin = self.Ein(-1*k)
        
        E1 = self.E1(Ein); h = self.ENactF(E1)#; En1 = self.E1(Enin); h_n = self.ENactF(En1)
        
        E2 = self.E2(h); h  = self.ENactF(E2)#; En2 = self.E2(h_n); h_n  = self.ENactF(En2)   
        
        E3 = self.E3(h); h = self.ENactF(E3)#; En3 = self.E3(h_n); h_n = self.actFlin(En3)
        
        #E4 = self.E4(h); h = self.ENactF(E4)#; En4 = self.E4(h_n); h_n = self.actFlin(En4)
        
        E = self.Eout(h)
        #E = self.Eout(h)
        #E_k = E + nfe
        
        #Lin = self.Lin(torch.cat((x,E_k),1))
        Lin = self.Lin(torch.cat((x,E),1))
        
        L1 = self.L1(Lin); h = self.normSine(L1)
        
        L2 = self.L2(h); h  = self.FNactF(L2)   
        
        L3 = self.L3(h); h = self.FNactF(L3)
        
        #L4 = self.L4(h); h = self.FNactF(L4)#h = self.actFlin(L4)
        
        U = self.out(h)

        #Ureal  = (U[:,0]).reshape(-1,1);Ui  = (U[:,0]).reshape(-1,1) # pass U out and fork later?
        
        #Ein = self.Ein(torch.cat((Ureal,k),1))


        extra = [nfe, E]

        return U, E, nfe, extra#return U, E_k, nfe, extra
        #return Ureal, Ui, E_k, extra
def evalModel(en, un, PATH):
    
    nn1 = evpNet(en,un)
    nn1.to(cuda)
    savepoint = torch.load(PATH)
    nn1.load_state_dict(savepoint['model_state_dict'])
    print('Loaded model at '+PATH)
    return nn1


def blochModel(eqParams, netParams,PATH, verbose = False, checkpoint=False):
    
    #eqParams = [b1,b2,k1d,k2d,V,period,npts,decayratio] netParams = [neurons, samplePoints, epochs, lr, minLoss, lrdecay]
    #            0  1   2   3  4   5     6      7                       0        1             2      3     4        5
    
    b1 = eqParams[0]; b2 = eqParams[1]; k1 = eqParams[2]; k2 = eqParams[3]; v = eqParams[4]; npts = eqParams[6]; decayratio = eqParams[7]
    
    print(eqParams)
    
    en_neurons = netParams[0]; fn_neurons = netParams[1]; points = netParams[2]; epochs = netParams[3]; lr = netParams[4]; minLoss = netParams[5]; lrdecay = netParams[6]; alphabeta = netParams[7]
    
    nn1 = evpNet(en_neurons,fn_neurons)
    nn1.to(cuda)
    nn2 = copy.deepcopy(nn1)

    # optimizer
    betas = [0.99, 0.999] #betas = [0.999, 0.9999] 
    optimizer = optim.Adam(nn1.parameters(), lr=0.0008, betas=betas)

    #scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs*decayratio)+1, gamma=lrdecay)
    
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=lrdecay, total_iters=int(epochs*decayratio)+1)
    
    loss = [] ; lossHistory = [] ; tHistory = [] ; EHistory = [] ; blochLossHistory = [] ; trivLossHistory = [] ; BCfLossHistory = [] ; BCdfLossHistory = [] ; trivfLossHistory = [] ; trivdfLossHistory = [] ; Residuals = [] ; nlossHistory = [] ; lrHistory = [] ; blochLossImaginaryHistory = [] ; BCimfLossHistory = [] ; BCimdfLossHistory = []
    tepochs = 0
    lossLim = 10e10 
    
    if checkpoint == True:
        savepoint = torch.load(PATH)
        nn1.load_state_dict(savepoint['model_state_dict'])
        #optimizer.load_state_dict(savepoint['optimizer_state_dict'])
        lossLim = savepoint['loss']
        
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
            
#####################################################################################################################################################
            
        resample = 1
        #alphabeta = 0.99
        
        
        kdist = Beta(torch.tensor([alphabeta]), torch.tensor([alphabeta]))
        if epoch % resample  == 0 and epoch != 0:

            kvec = ((k2-k1)*kdist.rsample((points,))+k1).to(cuda)

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
            
#####################################################################################################################################################
        FPswitch = 1000    
        
        if epoch > FPswitch or checkpoint == True:
            V = coulombPotential(grid, v, width, b1, npts)
        else:
            V = freeParticle(grid, v, width, b1, npts)
            
        NN, EK, NFE, ADJ = nn1(grid,k)
        
        EHistory.append(EK[0].clone())#EHistory.append(EK[0].cpu().data.numpy())
        
        if epoch % 1000 == 0 and verbose == True:
            print("u(x) MEAN:")
            print(NN.mean())
          
        # CALCULATE LOSS TERMS
        bloch_loss_real, bloch_loss_imaginary = blochLoss(NN, grid, EK, V, k)
        
        bc_loss_f, bc_loss_df = bcLoss(nn1, 100, 0, width, k1, k2) # network, point pairs, left bound, right bound, k1, k2
        bc_im_loss_f, bc_im_loss_df = bcImLoss(nn1, 100, 0, width, k1, k2)
        
        norm_loss = normLoss(nn1, 100, width, k1, k2, epoch)
        
        # OPTIM 
        lossTotal = bloch_loss_real + bloch_loss_imaginary + norm_loss + bc_loss_f + bc_loss_df + bc_im_loss_f + bc_im_loss_df 
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

        
        optimizer.zero_grad() # optimize 
    
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
        if  lossTotal < lossLim and epoch > FPswitch:# and epoch > epochs*0.5: # ADD CHECKPOINT CONDITION
            nn2 =  copy.deepcopy(nn1)
            lossLim=lossTotal

        # terminate training after loss threshold is reached 
        if lossTotal < minLoss and minLoss != -1 and epoch > FPswitch: # ADD CHECKPOINT CONDITION
            nn2 =  copy.deepcopy(nn1)
            print('Reached target loss')
            tepochs = epoch
            break
            
    time_end = time.time()
    runTime = time_end - time_start
    
    if tepochs == 0:
        tepochs = epochs
    # print metrics 
    print('Total runtime: ' + str(round(runTime,2)) + ' seconds')
    print('Final LR: '+ str(optimizer.param_groups[0]['lr']))
    print(str(round(tepochs/runTime,2)) + ' epochs/s')
    
    torch.save({
    'epoch': epoch,
    'model_state_dict': nn2.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': lossTotal,
    }, PATH)
    
    isthisloss = [lossHistory, blochLossHistory, BCfLossHistory, BCdfLossHistory, trivfLossHistory, trivdfLossHistory, lrHistory, blochLossImaginaryHistory, BCimfLossHistory, BCimdfLossHistory]
        
    return nn2, nn1, runTime, tepochs, isthisloss, tHistory, EHistory

############################################################################################################################
