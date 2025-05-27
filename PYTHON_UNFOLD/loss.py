import torch
import numpy as np

class SimplestLoss:
    @classmethod
    def forward(cls, beta, transfer):
        return torch.matmul(transfer, beta)

    @classmethod
    def loss(cls, x, transfer, reco, recoErr):
        fwd = cls.forward(x*reco, transfer)

        return torch.sum(torch.square((fwd-reco) / recoErr)) 
    
    @classmethod
    def nNuisances(cls):
        return 0

    @classmethod
    def t0(cls):
        return np.array([])

class SimpleWithBackgroundsLoss:
    @classmethod
    def genBkg(cls, beta, gamma):
        return beta * torch.square(gamma[0]) + torch.square(gamma[1])

    @classmethod
    def recoBkg(cls, p, rho):
        return p * torch.square(rho[0]) + torch.square(rho[1])

    @classmethod
    def forward(cls, beta, transfer, gamma, rho):
        genpure = beta - cls.genBkg(beta, gamma)
        p = torch.matmul(transfer, genpure)
        return p + cls.recoBkg(p, rho)

    @classmethod
    def loss(cls, x, transfer, reco, recoErr):
        beta = x[:-4]
        gamma = x[-4:-2]
        rho = x[-2:]

        fwd = cls.forward(beta*reco, transfer, gamma, rho)

        errTerm = torch.sum(torch.square((fwd-reco) / recoErr))
        cstrTermG = torch.sum(torch.square(gamma))
        cstrTermR = torch.sum(torch.square(rho))
        return errTerm + cstrTermG + cstrTermR

    @classmethod
    def nNuisances(cls):
        return 4

    @classmethod
    def t0(cls):
        return np.array([0.0, 0.0, 0.0, 0.0])

class SimpleFullModelLoss:
    @classmethod
    def genBkg(cls, beta, gamma):
        return beta * (1e-3 + gamma[0]*1e-4) + (1e-3 + gamma[1]*1e-4)

    @classmethod
    def recoBkg(cls, p, rho):
        return p * torch.square(rho[0]) + torch.square(rho[1])

    @classmethod
    def forward(cls, beta, transfer, theta, gamma, rho):
        genpure = beta - cls.genBkg(beta, gamma)

        thetransfer = transfer[0] + torch.tensordot(theta, transfer[1:], 1)

        p = torch.matmul(thetransfer, genpure)
        return p + cls.recoBkg(p, rho)

    @classmethod
    def loss(cls, x, transfer, reco, recoErr):
        beta = x[:-6]
        theta = x[-6:-4]
        gamma = x[-4:-2]
        rho = x[-2:]

        fwd = cls.forward(beta*reco, transfer, theta, gamma, rho)

        errTerm = torch.sum(torch.square((fwd-reco) / recoErr))
        cstrTermG = torch.sum(torch.square(gamma))
        cstrTermR = torch.sum(torch.square(rho))
        cstrTermT = torch.sum(torch.square(theta))
        return errTerm + cstrTermG + cstrTermR + cstrTermT

    @classmethod
    def nNuisances(cls):
        return 6

    @classmethod
    def t0(cls):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

class SimpleFullModelFullTemplateLoss:
    def __init__(self, transfer, gamma0, gammaErr, rho0, rhoErr):
        self.transfer = transfer
        self.gamma0 = gamma0
        self.gammaErr = gammaErr
        self.rho0 = rho0
        self.rhoErr = rhoErr

        self.nGamma = gamma0.shape[0]
        self.nRho = rho0.shape[0]
        self.nTheta = transfer.shape[0] - 1
        self.nBeta = transfer.shape[2]

        print("nBeta:", self.nBeta)
        print("nTheta:", self.nTheta)
        print("nGamma:", self.nGamma)
        print("nRho:", self.nRho)

    def genBkg(self, beta, gamma):
        return beta * (self.gamma0 + gamma*self.gammaErr)

    def recoBkg(self, p, rho):
        return p * (self.rho0 + rho*self.rhoErr)

    def forward(self, beta, theta, gamma, rho):
        genpure = beta - self.genBkg(beta, gamma)

        thetransfer = self.transfer[0] + torch.tensordot(theta, self.transfer[1:], 1)

        p = torch.matmul(thetransfer, genpure)
        return p + self.recoBkg(p, rho)

    def loss(self, x, reco, recoErr):
        beta = x[:self.nBeta]
        theta = x[self.nBeta:self.nBeta+self.nTheta]
        gamma = x[self.nBeta+self.nTheta:self.nBeta+self.nTheta+self.nGamma]
        rho = x[self.nBeta+self.nTheta+self.nGamma:]

        fwd = self.forward(beta*reco, theta, gamma, rho)

        errTerm = torch.sum(torch.square((fwd-reco) / recoErr))
        cstrTerm = torch.sum(torch.square(x[self.nBeta:]))
        return 0.5 * (errTerm + cstrTerm)

    def one_parameter_loss(self, reco, recoErr):
        return lambda x: self.loss(x, reco, recoErr)

    def nNuisances(self):
        return self.nGamma + self.nRho + self.nTheta

    def cuda(self):
        self.transfer = self.transfer.cuda()
        self.gamma0 = self.gamma0.cuda()
        self.gammaErr = self.gammaErr.cuda()
        self.rho0 = self.rho0.cuda()
        self.rhoErr = self.rhoErr.cuda()

        return self

class FullLoss:
    '''
    This is the full Loss model we use. The forward model works as follows:
        1. Gen-level background = G * gen
        2. Detector model applies to background-free gen component to obtain background-free reco:
            pureReco = T * (gen - gen background)
        3. Reco-level background = R * pureReco
        4. Return reco = pureReco + reco background

    We have three qualitatively different kinds of systematics:
        1. Variations in the transfer matrix. These are parameterized as 
            T = transfer0 + sum_i theta_i transferVariation_i
        2. Shape variations in the gen-level background template
            G = gamma0 + sum_i gamma_i gammaVariation_i
        3. Shape variations in the reco-level background template
            R = rho0 + sum_i rho_i rhoVariations_i

    All three have gaussian constraints,

    The loss a simple chi-squared loss function:

    L = 1/2 (reco_pred - reco_true)^2 / reco_uncertainty^2 + 1/2 sum_i (nuisance_i)^2
    TODO: is the 1/2 correct?
    '''
    def __init__(self, transfer0, transferVariations, gamma0, gammaVariations, rho0, rhoVariations):
        self.transfer0 = torch.from_numpy(transfer0)
        self.transferVariations = torch.from_numpy(transferVariations)
        self.gamma0 = torch.from_numpy(gamma0)
        self.gammaVariations = torch.from_numpy(gammaVariations)
        self.rho0 = torch.from_numpy(rho0)
        self.rhoVariations = torch.from_numpy(rhoVariations)

        self.nGamma = gammaVariations.shape[0]
        self.nRho = rhoVariations.shape[0]
        self.nTheta = transferVariations.shape[0]
        self.nBeta = transfer0.shape[1]

        print("nBeta:", self.nBeta)
        print("nTheta:", self.nTheta)
        print("nGamma:", self.nGamma)
        print("nRho:", self.nRho)

    def genBkg(self, beta, gamma):
        G = self.gamma0 + torch.tensordot(gamma, self.gammaVariations, 1)
        return G * beta

    def recoBkg(self, p, rho):
        R = self.rho0 + torch.tensordot(rho, self.rhoVariations, 1)
        return R * p

    def forward(self, beta, theta, gamma, rho):
        genpure = beta - self.genBkg(beta, gamma)

        thetransfer = self.transfer0 + torch.tensordot(theta, self.transferVariations, 1)

        p = torch.matmul(thetransfer, genpure)

        return p + self.recoBkg(p, rho)

    def loss(self, x, reco, recoErr):
        beta = x[:self.nBeta]
        theta = x[self.nBeta:self.nBeta+self.nTheta]
        gamma = x[self.nBeta+self.nTheta:self.nBeta+self.nTheta+self.nGamma]
        rho = x[self.nBeta+self.nTheta+self.nGamma:]

        fwd = self.forward(beta*reco, theta, gamma, rho)

        errTerm = torch.sum(torch.square((fwd-reco) / recoErr))
        cstrTerm = torch.sum(torch.square(x[self.nBeta:]))
        return 0.5 * (errTerm + cstrTerm)

    def one_parameter_loss(self, reco, recoErr):
        theerr = torch.where(recoErr == 0, 1, recoErr)
        return lambda x: self.loss(x, reco, theerr)

    def nNuisances(self):
        return self.nGamma + self.nRho + self.nTheta

    def cuda(self):
        self.transfer0 = self.transfer0.cuda()
        self.gamma0 = self.gamma0.cuda()
        self.rho0 = self.rho0.cuda()
        self.transferVariations = self.transferVariations.cuda()
        self.gammaVariations = self.gammaVariations.cuda()
        self.rhoVariations = self.rhoVariations.cuda()

        return self

losses = {
    "Simplest" : SimplestLoss,
    "SimpleWithBackgrounds" : SimpleWithBackgroundsLoss,
    "SimpleFullModel" : SimpleFullModelLoss
}
