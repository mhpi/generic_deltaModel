import copy
import torch
import torch.nn as nn
import sourcedefender
from solver.batchJacobian import batchJacobian


# torch.autograd.set_detect_anomaly(True)
def batchJacobian_FD(x, G, epsilon):
    nb, nx = x.shape
    ny = nx  # Since we are perturbing x, ny is the number of features in x

    # Create a perturbed version of x
    xE = x.repeat_interleave(ny + 1, dim=0).double()

    perturbation = torch.eye(ny).unsqueeze(0).expand(nb, -1, -1) * epsilon
    perturbation = torch.cat([torch.zeros(nb, 1, ny), perturbation], dim=1).to(x)

    # Expand x and add perturbations
    xE = xE + perturbation.reshape(nb * (ny + 1), ny)
    xE = xE.reshape(nb, ny+1 , ny).unsqueeze(1)

    G = G.G_extend(ny)
    # Compute G for all perturbed inputs in one run
    ggE = G(xE)# .view(nb, ny + 1, nx)

    # Extract the original and perturbed G values
    gg_original = ggE[:,:, 0, :]
    gg_perturbed= ggE[:,:,1:,:]

    # Compute finite differences for the Jacobian
    dGdx = ((gg_perturbed - gg_original.unsqueeze(2)) / epsilon).permute(0,1,3,2)#.permute(0, 2, 1)

    return dGdx, gg_original


class Jacobian(nn.Module):
    # DescriptionL an wrapper for all the Jacobian options -- stores some options and provide a uniform interface
    # J=Jacobian((mtd="batchScalarJacobian_AD"),(func))
    # jac=J(x,y)
    # x can be a torch.Tensor, or a tuple/list of Tensors, in which case the return will be a tuple/list of Tensors
    def __init__(self, mtd=0, create_graph=True, epsilon = 1.e-8):# 1e-2 # 6.0555e-06
        super(Jacobian, self).__init__()
        self.mtd = mtd
        self.create_graph = create_graph
        self.epsilon = epsilon

    def forward(self, x, y = None, G = None):
        # adaptively select the right function
        if self.mtd == 0 or self.mtd == "batchJacobianAD":
            Jac = batchJacobian(y, x, graphed=self.create_graph, batchx = True)
        elif self.mtd == 1 or self.mtd == "batchJacobianFD":
            Jac = batchJacobian_FD(x , G = G, epsilon = self.epsilon)
        else:
            raise ValueError("Please choose a valid Jacobian class")
        return Jac


def rtnobnd(ctx, x0, G, J, settings, doPrint=False, eval = False):
    # Description: solves the nonlinear problem with unbounded Newton iteration
    # may have poor global convergence. but if it works for your function it should be fast.
    x    = x0.clone()
    nx   = 1 if x.ndim == 1 else x.shape[-1]
    iter = 0
    xtol_check = True
    ftol_check = True


    while (iter < settings["maxiter"]) and (ftol_check) and (xtol_check):
        # torch.set_grad_enabled(True)
        # x.requires_grad = True
        f = G(x)

        if torch.isnan(f).any():
            print("True")
            break

        dfdx = J(x, f.view(-1, x.shape[-1]))#J(x, f)
        # x = x.detach()
        # torch.set_grad_enabled(False)

        if nx == 1:
            xnew = x - f / dfdx
        else:
            deltaX = torch.linalg.solve(dfdx, f.view(x.shape))
            xnew = x - deltaX
        xtol_check, ftol_check, ftol =  G.check_tol(x, xnew, f, settings)
        x = xnew
        iter += 1
        if doPrint:
            print(f"iter={iter},ftol= {ftol}")

    # torch.set_grad_enabled(True)
    # x = x.detach()
    # if not eval:
    #     dfdp = J( G.p,f.view(-1, x.shape[-1]))#J(f.view(-1, x.shape[-1]), G.p, graphed=True)
    #
    #     if torch.isnan(dfdp).any() or torch.isinf(dfdp).any():
    #         raise RuntimeError(f"Jacobian matrix is NaN")
    #
    #     ctx.save_for_backward(dfdp, dfdx)
    #
    # torch.set_grad_enabled(False)
    del f
    return x

def rtFD(x0, G, J, settings):
    torch.set_grad_enabled(False)
    x = x0.double().clone().detach()
    iter = 0

    dGdx, gg = J(x.squeeze(1), G = G)

    if torch.isnan(dGdx).any() or torch.isinf(dGdx).any():
        raise RuntimeError(f"Jacobian matrix is NaN")

    resnorm = torch.linalg.norm(gg, float('inf'), dim=[2])  # calculate norm of the residuals
    resnorm0 = 100 * resnorm
    xtol_check = True
    ftol_check = True

    # while ((torch.max(resnorm) > settings['ftol']) and iter <= settings['maxiter']):
    while (iter < settings["maxiter"]) and (ftol_check) and (xtol_check):

        iter += 1
        if torch.max(resnorm / resnorm0) > 0.2:

            dGdx, gg = J(x.squeeze(1), G = G)
            if torch.isnan(dGdx).any() or torch.isinf(dGdx).any():
                raise RuntimeError(f"Jacobian matrix is NaN")

        if dGdx.ndim == gg.ndim:  # same dimension, must be scalar.
            dx = (gg / dGdx)
        else:
            dx = torch.linalg.solve(dGdx, gg)# matrixSolve(dGdx, gg)
        xnew = x - dx
        gg = G(xnew)
        resnorm0 = resnorm;  ##% old resnorm
        resnorm = torch.linalg.norm(gg, float('inf'), dim=[2])

        xtol_check, ftol_check, ftol =  G.check_tol(x, xnew, gg, settings)
        x = xnew

    # This way, we reduced one forward run. You can also save these two to the CPU if forward run is
    # Alternatively, if memory is a problem, save x and run g during the backward.

    torch.set_grad_enabled(True)
    x = x.detach()
    # if not eval:
    #     dfdp = batchJacobian(f, G.p, graphed=True)
    #
    #     if torch.isnan(dfdp).any() or torch.isinf(dfdp).any():
    #         raise RuntimeError(f"Jacobian matrix is NaN")
    #
    #     ctx.save_for_backward(dfdp, dfdx)
    #
    # torch.set_grad_enabled(False)
    del gg
    # print("resnorm", torch.max(resnorm).item())
    # print("iterations", iter)
    return x



class tensorNewton(nn.Module): #torch.autograd.Function
    # Description: solve a nonlinear problem of G(x)=0 using Newton's iteration
    # x can be a vector of unknowns. [nb, nx] where nx is the number of unknowns and nb is the minibatch size
    # minibatch is for different sites, physical parameter sets, etc.
    # model(x) should produce the residual
    def __init__(self, G, J=Jacobian(), mtd=0, lb=None, ub=None,settings={"maxiter": 10, "ftol": 1e-6, "xtol": 1e-6, "alpha": 0.75}):
        # alpha, the gradient attenuation factor, is only for some algorithms.
        super(tensorNewton, self).__init__()
        self.G = G
        self.J = J
        self.mtd = mtd
        self.lb = lb
        self.ub = ub
        self.settings = settings

    def forward(self, x0):
        if self.mtd == 0 or self.mtd == 'rtnobnd' or self.mtd == "batchJacobianAD":
            return rtnobnd(self, x0, self.G, self.J, self.settings)
        elif self.mtd == 1 or self.mtd == "rtFD" or self.mtd == "batchJacobianFD":
            return rtFD(x0, self.G, self.J, self.settings)
        else:
            assert self.mtd <= 1, 'tensorNewton:: the nonlinear solver has not been implemented yet'

        # @staticmethod
    # def forward(ctx, x0, G, J=None, mtd=0, settings = None):
    #     if J is None:
    #         J = Jacobian()  # Default Jacobian initialization
    #     if settings is None:
    #         settings = {"maxiter": 10, "ftol": 1e-6, "xtol": 1e-6, "alpha": 0.75}
    #
    #     if mtd == 0 or mtd == 'rtnobnd' or mtd == "batchJacobianAD":
    #         return rtnobnd(ctx, x0, G, J, settings)
    #     elif mtd == 1 or mtd == "rtFD" or mtd == "batchJacobianFD":
    #         return rtFD(ctx, x0, G, J, settings)
    #     else:
    #         assert mtd <= 1, 'tensorNewton:: the nonlinear solver has not been implemented yet'
    #
    # def backward(ctx):
    #     pass
        return


class nonlinearsolver(torch.nn.Module):
    def __init__(self, PBM, c_forcing,wrf_plant,wkf_plant,wrf_soil, wkf_soil, hydro_media_id, dtime,  nly, mtd, p= None):
        super(nonlinearsolver, self).__init__()

        self.PBM       = PBM
        self.wrf_plant = wrf_plant
        self.wkf_plant = wkf_plant
        self.wrf_soil  = wrf_soil
        self.wkf_soil  = wkf_soil
        self.hydro_media_id = hydro_media_id
        self.dtime     = dtime
        self.nly       = nly
        self.c_forcing = c_forcing
        self.mtd       = mtd
        self.p         = p

    # output = model.forward(input) # where input are for the known data points
    def forward(self, x):

        x = x.clone()
        nly = self.nly
        ci, veg_tempk, th_node = x[..., 0:nly], x[..., nly:2*nly], x[...,2*nly:]
        f = self.PBM.solve_for_Q_th(ci,veg_tempk, th_node, self.c_forcing,
                                    self.wrf_plant, self.wkf_plant, self.wrf_soil, self.wkf_soil,
                                    self.hydro_media_id, self.dtime,self.mtd, self.p)

        return f

    def check_tol(self, x, xnew, f, settings):
        id1 = self.nly;id2 = 2 * id1
        ftol = f.abs().max()
        xtol = f.abs().max()

        if settings['Model']==0:
            ftol1 = f[...,0  :id1].abs().max()
            ftol2 = f[...,id1:id2].abs().max()
            ftol3 = f[...,id2:].abs().max()
            ftol_check = (ftol1 > settings['ftol1']) | (ftol2 > settings['ftol2'])| (ftol3 > settings['ftol3'])
        else:
            ftol_check = ftol > settings["ftol"]

        if settings['Model']==0:
            xtol1 = (xnew[...,0  :id1] - x[...,0  :id1]).abs().max()
            xtol2 = (xnew[...,id1:id2] - x[...,id1:id2]).abs().max()
            xtol3 = (xnew[...,id2:]    - x[...,id2:]).abs().max()
            xtol_check = (xtol1 > settings['xtol1']) | (xtol2 > settings['xtol2']) | (xtol3 > settings['xtol3'])
        else:
            xtol_check = xtol > settings["xtol"]
        return xtol_check, ftol_check, ftol
    def G_extend(self, ny):
        G = copy.deepcopy(self)
        G.c_forcing       = G.c_forcing.unsqueeze(dim=2).repeat_interleave(repeats = ny+1, dim = 2)
        G.PBM.params      = G.PBM.params.unsqueeze(dim=2).repeat_interleave(repeats = ny+1, dim = 2)
        G.PBM.attributes  = G.PBM.attributes.unsqueeze(dim=2).repeat_interleave(repeats = ny+1, dim = 2)
        G.PBM.mstates     = {key: torch.repeat_interleave(value.unsqueeze(dim=2), repeats=ny+1, dim=2) for key, value in G.PBM.mstates.items()}
        G.PBM.mvars       = {key: torch.repeat_interleave(value.unsqueeze(dim=2), repeats=ny+1, dim=2) for key, value in G.PBM.mvars.items()}
        G.wrf_soil.attrs  = {key: torch.repeat_interleave(value.unsqueeze(dim=2), repeats=ny+1, dim=2) for key, value in G.wrf_soil.attrs.items()}
        G.wkf_soil.attrs  = {key: torch.repeat_interleave(value.unsqueeze(dim=2), repeats=ny+1, dim=2) for key, value in G.wkf_soil.attrs.items()}
        G.wrf_plant.attrs = {key: torch.repeat_interleave(value.unsqueeze(dim=2), repeats=ny+1, dim=2) for key, value in G.wrf_plant.attrs.items()}
        G.wkf_plant['plant_media'].attrs = {key: torch.repeat_interleave(value.unsqueeze(dim=2), repeats=ny+1, dim=2) for key, value in G.wkf_plant['plant_media'].attrs.items()}

        G.wrf_soil.updateAttrs(G.wrf_soil.attrs)
        G.wkf_soil.updateAttrs(G.wkf_soil.attrs )
        G.wrf_plant.updateAttrs(G.wrf_plant.attrs )
        G.wkf_plant['plant_media'].updateAttrs(G.wkf_plant['plant_media'].attrs )

        if G.p is not None:
            G.p = G.p.unsqueeze(dim=2)
        return G

