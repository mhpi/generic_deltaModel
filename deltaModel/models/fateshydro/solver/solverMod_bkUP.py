import copy
import torch
import torch.nn as nn
import sourcedefender
from solver.batchJacobian import batchJacobian


# torch.autograd.set_detect_anomaly(True)
def batchScalarJacobian_AD(x, y, graphed=True):
    # Description: extract the gradient dy/dx for scalar y (but with minibatch)
    # relying on the fact that the minibatch has nothing to do with each other!
    # y: [nb]; x: [nb, (?nx)]. For single-column x, get dydx as a vector [dy1/dx1,dy2/dx2,...,dyn/dxn]; For multiple-column x, there will be multi-column output
    # output: [nb, (nx?)]
    # if x is tuple/list, the return will also be tuple/list
    assert not (y.ndim > 1 and y.shape[-1] > 1), 'this function is only valid for batched scalar y outputs'
    gO = torch.ones_like(y, requires_grad=False)  # donno why it does not do 2D output
    dydx = torch.autograd.grad(outputs=y, inputs=x, retain_graph=True, grad_outputs=gO, create_graph=graphed)
    # calculate vjp. For the minibatch, we are taking advantage of the fact that the y at a site is unrelated to
    # the x at another site, so the matrix multiplication [1,1,...,1]*J reverts to extracting the diagonal of the Jacobian
    if isinstance(x, torch.Tensor):
        dydx = dydx[0]  # it gives a tuple
    if not graphed:
        # during test, we detach the graph
        # without doing this, the following cannot be cleaned from memory between time steps as something use them outside
        if isinstance(dydx, torch.Tensor):
            dydx = dydx.detach()
        else:
            for dd in dydx:
                dd = dd.detach()
        y = y.detach()
        gO = gO.detach()
    return dydx

def batchJacobian_AD(x, y, graphed=True, doSqueeze=True):
    # Desription: extract the jacobian dy/dx for multi-column y output (and with minibatch)
    # compared to the scalar version above, this version will call grad() ny times and store outputs in a tensor matrix
    # y: [nb, ny]; x: [nb, nx]. x could also be a tuple or list of tensors.
    # permute and view your y to be of the above format.
    # AD jacobian is not free and may end up costing me time
    # output: Jacobian [nb, ny, nx] # will squeeze after the calculation
    # relfying on the fact that the minibatch has nothing to do with each other!
    # if they do, i.e, they come from different time steps of a simulation, you need to put them in second dim in y!
    # view or reshape your x and y to be in this format if they are not!
    # pay attention, this operation could be expensive.
    ny = 1 if y.ndim == 1 else y.shape[-1]

    #=========================================

    #=========================================
    # prepare the receptacle
    if isinstance(x, torch.Tensor):
        nx = 1 if x.ndim == 1 else x.shape[-1]
        sizes = [y.shape[0], ny, nx]
        DYDX0 = torch.zeros(sizes, requires_grad=True, device = y.device, dtype = y.dtype)
        DYDX = DYDX0.clone()  # avoid the "leaf variable cannot have in-place operation" issue
    elif isinstance(x, tuple) or isinstance(x, list):
        # prepare a list of tensors
        DYDX = list()
        for i in range(0, len(x)):
            nx = x[i].shape[-1]
            sizes = [y.shape[0], ny, nx]
            DYDX0 = torch.zeros(sizes, requires_grad=True)
            DYDX.append(DYDX0.clone())  # avoid the "leaf variable cannot have in-place operation" issue

    gO = torch.ones_like(y[:, 0], requires_grad=False)  # donno why it does not do 2D output
    for i in range(ny):
        dydx = torch.autograd.grad(outputs=y[:, i], inputs=x, retain_graph=True, grad_outputs=gO, create_graph=graphed)
        if isinstance(x, torch.Tensor):
            #===================================10TH MAY 2024
            DYDX[:, i, :] = dydx[0].view(-1,nx)  # for some reason it gives me a tuple
            #===================================
            if doSqueeze:
                DYDX = DYDX.squeeze()
        elif isinstance(x, tuple) or isinstance(x, list):
            for j in range(len(x)):
                DYDX[j][:, i, :] = dydx[j]  # for some reason it gives me a tuple
    # dydx2 = torch.autograd.grad(outputs=y[:,i], inputs=x, retain_graph=True, grad_outputs=gO, create_graph=graphed, is_grads_batched=True)

    if not graphed:
        # during test, we may detach the graph
        # without doing this, the following cannot be cleaned from memory between time steps as something use them outside
        # however, if you are using the gradient during test, then graphed should be false.
        dydx = dydx.detach()
        DYDX0 = DYDX0.detach()
        DYDX = DYDX.detach()
        x = x.detach()
        y = y.detach()
        gO = gO.detach()
    return DYDX

def getDictJac(xDict, yDict, J, rt=1, xfs=("params", "u0"), yfs=("yP",), ySel=([],), yPermuteDim=([],)):
    # Description: provides a succinct way of extracting Jacobian from a dictionary. Gives an interface to select indices along dimensions and permute.
    # JAC[i][j] is for yfs[i] and xfs[j]
    # NAMES describes these entries.
    # ySel is the tuple specifying selection (dim, tensor(index)). indexing the original data without changing number of dimensions.
    # permuteDim: if non-empty, will permute using this argument. permute happens after indexing
    X = list();
    nx = 1;
    for xf in xfs:
        dat = xDict[xf]
        X.append(dat)  # it needs to append the whole thing, not a slice.
        if dat.ndim > 1:
            nx = nx * dat.shape[-1]

    JAC = list()
    NAMES = list()
    for yf, ys, pd in zip(yfs, ySel, yPermuteDim):
        d = yDict[yf]
        # d = selectDims(dat, (d0,d1,d2)).view([dat.shape[0], int(dat.nelement()/dat.shape[0])]) # essentially dat[:,d1,d2], but [] means select the entire axis. also safe with 1D/2D arrays.
        if len(ys) == 2:
            d = torch.select(d, ys[0], ys[1])
        if len(pd) > 0:
            d = torch.permute(d, pd)
        # d = torch.squeeze(d) # remove singleton dimension

        jac0 = J(X, d)
        JAC.append(jac0)

        names = list()
        for xf in xfs:
            names.append(f"d({yf}{ySel}{yPermuteDim})/d({xf})")
        NAMES.append(names)
    return JAC, NAMES


def batchScalarJacobian_FD(x, y, func, dxScalar, lb, ub):
    # Description: finite difference -- extracting the jacobian dy/dx for scalar x (but with minibatch)
    # y: [nb]; x: [nb]. # scalars. nb is minibatch dimension. Must require there is no connection between the nb data points
    # dx is the fraction between lb and ub
    dxf = torch.ones_like(x, requires_grad=True)
    prec = 1e-6
    mask = (x + prec) < ub
    dxf = dxf * mask * dxScalar - dxf * (~mask) * dxScalar
    dx = (ub - lb) * dxf
    xnew = x + dx
    y2 = func(xnew)
    dydx = (y2 - y) / dx
    return dydx

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
    def __init__(self, mtd=0, func=None, create_graph=True, settings={"dx": 1e-2}, epsilon = 1.e-8):# 1e-2 # 6.0555e-06
        super(Jacobian, self).__init__()
        self.mtd = mtd
        self.func = func
        self.settings = settings
        self.create_graph = create_graph
        self.epsilon = epsilon

    def forward(self, x, y = None, G = None):
        if y is not None:
            ny = 1 if y.ndim == 1 else y.shape[-1]
        # adaptively select the right function
        if self.mtd == 0 or self.mtd == "batchScalarJacobian_AD":  # we can also add or ny==1
            Jac = batchScalarJacobian_AD(x, y, graphed=self.create_graph)
        elif self.mtd == 1 or self.mtd == "batchJacobian_AD":
            Jac = batchJacobian_AD(x, y, graphed=self.create_graph, doSqueeze=False)
        elif self.mtd == 2 or self.mtd == "batchScalarJacobian_FD":
            dxScalar = self.settings["dx"]
            lb, ub = self.settings["bounds"]
            func = self.func
            Jac = batchScalarJacobian_FD(x, y, func, dxScalar, lb, ub)
        elif self.mtd == 3 or self.mtd == "batchJacobian":
            Jac = batchJacobian(y, x, graphed=self.create_graph, batchx = True)
        elif self.mtd == 4 or self.mtd == "batchJacobian_FD":
            Jac = batchJacobian_FD(x , G = G, epsilon = self.epsilon)
        else:
            raise ValueError("Please choose a valid Jacobian class")
        return Jac


def rtnobnd(x0, G, J, settings, doPrint=False):
    # Description: solves the nonlinear problem with unbounded Newton iteration
    # may have poor global convergence. but if it works for your function it should be fast.
    x    = x0.clone()
    nx   = 1 if x.ndim == 1 else x.shape[-1]
    iter = 0
    xtol_check = True
    ftol_check = True

    while (iter < settings["maxiter"]) and (ftol_check) and (xtol_check):
        f = G(x)

        if torch.isnan(f).any():
            print("True")
            break

        dfdx = J(x, f.view(-1, x.shape[-1]))#J(x, f)
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
    return x

def rtFD(x0, G, J, settings):

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
    del gg
    # print("resnorm", torch.max(resnorm).item())
    # print("iterations", iter)
    return x


def rtsafe(x0, G, J, lb, ub, settings, doPrint=False):
    # Description: safe newton iteration with bounds --- trial evaluations won't exceed bounds
    # mixed algorithm between newton's and midpoint
    # modified from numerical recipes http://phys.uri.edu/nigh/NumRec/bookfpdf/f9-4.pdf
    # also in PAWS. https://bitbucket.org/lbl-climate-dev/psu-paws-git/src/master/pawsPack/PAWS/src/PAWS/vdata.f90
    # solves the nonlinear problem with a range bound
    # x: [nb]
    iter = 0
    ftol = 1e3
    xtol = 1e4
    prec = 1e-10
    nx = 1 if x0.ndim == 1 else x0.shape[-1]
    x1 = torch.zeros_like(x0) + lb
    x2 = torch.zeros_like(x0) + ub
    alphaScalar = settings["alpha"]
    maxiter = settings["maxiter"]
    ftol_crit = settings["ftol"]
    xtol_crit = settings["xtol"]

    with torch.no_grad():
        # these selections do not need to be tracked, as long as the results do not
        # participate in actual computations.
        fl = G(x1)
        fh = G(x2)

    mask = fl < 0.0  # tensor can be used as numerical values
    xl = x1 * mask + (~mask) * x2
    xh = x2 * mask + (~mask) * x1

    mask = (x0 > x1) & (x0 < x2)
    x0 = (mask * x0 + (~mask) * 0.5 * (x1 + x2)).requires_grad_()
    x = x0.clone()  # avoids the leaf variable issue, and it has to be done this way -- cannot clone first and requires_grad
    f = G(x)
    dfdx = J(x, f)

    maskNull = fl * fh > 0
    dxOld = x2 - x1;
    dx = dxOld.clone();
    fOld = f.clone();

    for iter in range(maxiter):

        mask1 = f * fOld < -prec  # detect oscillations
        alpha = mask1 * alphaScalar + (~mask1) * 1.0  # attenuate the gradient to dampen oscillations
        xnew = x - alpha * f / dfdx
        mask = (((x - xh) * dfdx - f) * ((x - xl) * dfdx - f) > 0.0) | ((2.0 * f).abs() > (dxOld * dfdx).abs()) \
               | (torch.isnan(xnew)) | (dfdx.abs() < prec)
        xnewmid = 0.5 * (xl + xh)
        xnew[mask] = xnewmid[
            mask]  # doing ordinary mask addition does not work because NaN interacting with anything still produces NaN
        dxOld.copy_(dx)
        fOld.copy_(f)

        dx = xnew - x
        f  = G(xnew);
        dfdx = J(xnew, f)
        ftol = f.abs().max()
        xtol = dx.abs().max()
        x.copy_(xnew)
        mask2 = f < 0
        xl = mask2 * x + (~mask2) * xl
        xh = (~mask2) * x + mask2 * xh

        ##added
        # if torch.isnan(f).any():## for anet< 0 it gives nan stomatal conductance so we should break the loop
        #     print("True")
        #     break
        ##added

        iter += 1
        if doPrint:
            print(
                f"iter={iter}, x= {float(x[1])}, y= {float(f[1])}, dfdx= {float(dfdx[1])}, xtol= {xtol}, ftol= {ftol}")
        isConverged = (ftol < ftol_crit) | (xtol < xtol_crit)
        if isConverged:  # put this here rather a loop as maybe ftol will be NaN
            break

    x[maskNull] = 1e20  # root is not bounded.
    return x


class tensorNewton(nn.Module):
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
        if self.mtd == 0 or self.mtd == 'rtnobnd':
            return rtnobnd(x0, self.G, self.J, self.settings)
        elif self.mtd == 1 or self.mtd == 'rtsafe':
            assert self.lb is not None, 'mtd==1, using bounded rtsafe, but no upper bound is provided!'
            return rtsafe(x0, self.G, self.J, self.lb, self.ub, self.settings)
        elif self.mtd == 2 or self.mtd == "rtFD":
            return rtFD(x0, self.G, self.J, self.settings)
        else:
            assert self.mtd <= 2, 'tensorNewton:: the nonlinear solver has not been implemented yet'


class nonlinearsolver(torch.nn.Module):
    def __init__(self, PBM, c_forcing,wrf_plant,wkf_plant,wrf_soil, wkf_soil, hydro_media_id, dtime,  nly, mtd, pDyn= None):
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
        self.pDyn      = pDyn

    # output = model.forward(input) # where input are for the known data points
    def forward(self, x):
        x = x.clone()
        nly = self.nly
        ci, veg_tempk, th_node = x[..., 0:nly], x[..., nly:2*nly], x[...,2*nly:]
        f = self.PBM.solve_for_Q_th(ci,veg_tempk, th_node, self.c_forcing,
                                    self.wrf_plant, self.wkf_plant, self.wrf_soil, self.wkf_soil,
                                    self.hydro_media_id, self.dtime,self.mtd, self.pDyn)

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

        if G.pDyn is not None:
            G.pDyn = G.pDyn.unsqueeze(dim=2)
        return G
# class testsolver(torch.nn.Module):
#     def __init__(self, a):
#         super(testsolver, self).__init__()
#         self.a = a
#
#     # output = model.forward(input) # where input are for the known data points
#     def forward(self, x):
#         x = x.clone()
#         f =x**2 - 3*x + self.a
#         nsites, ncohorts, nodes, _ = f.shape
#         f = f.view(-1, nodes)
#         return f
#
# class thnodeSolver(torch.nn.Module):
#     def __init__(self, PBM, wrf_plant,wkf_plant,wrf_soil, wkf_soil, dtime, qcanopy, mtd):
#         super(thnodeSolver, self).__init__()
#
#         self.PBM       = PBM
#         self.wrf_plant = wrf_plant
#         self.wkf_plant = wkf_plant
#         self.wrf_soil  = wrf_soil
#         self.wkf_soil  = wkf_soil
#         self.dtime     = dtime
#         self.qcanopy   = qcanopy
#         self.mtd       = mtd
#
#     # output = model.forward(input) # where input are for the known data points
#     def forward(self, x):
#         x = x.clone()
#         f = self.PBM.solve_for_th(self.wrf_plant, self.wkf_plant,
#                                   self.wrf_soil , self.wkf_soil , x, self.dtime, self.qcanopy, self.mtd)
#
#         return f
# class ci_tleaf_solver(torch.nn.Module):
#     # Description: Includes our nonlinear system for the leaflayer photosynthesis subroutine in
#     # the photosynthesis module in FATES
#
#     # Inputs                  :Forcing dataset which includes:
#     # gb_mol                  :leaf boundary layer conductance (umol H2O/m**2/s)
#     # can_press               :Air pressure NEAR the surface of the leaf (Pa)
#     # can_co2_ppress          :Partial pressure of CO2 NEAR the leaf surface (Pa)
#     def __init__(self,PBM, c_forcing, mtd):
#         super(ci_tleaf_solver, self).__init__()
#         self.PBM        = PBM
#         self.c_forcing  = c_forcing
#         self.mtd        = mtd
#
#
#     def forward(self, x):
#         x = x.clone()
#         # x = x.view(self.attributes['lai'].shape+(2,))
#         # start = time.time()
#         ci, veg_tempk = x[...,0], x[...,1]
#         f = self.PBM.solve_for_Q( ci, veg_tempk,self.c_forcing, self.mtd)
#
#         return f
# def testJacobian():
#     a = torch.tensor([[
#     [
#         [1, 1.5, 0.5],
#         [2, 0.75, 2.25],
#         [1.8, 1.2, 2]
#     ],
#     [
#         [0.9, 1.1, 0.4],
#         [1.3, 2, 1.5],
#         [0.8, 0.6, 2.1]
#     ],
#     # [
#     #     [1.7, 0.3, 0.2],
#     #     [2.2, 1.9, 0.1],
#     #     [1, 1.4, 2.25]
#     # ]
# ]])
#     x0 = torch.tensor([[[[2.0, 2.0 , 2.0],
#                         [2.0, 2.0 , 1.0],
#                         [2.0, 2.0 , 2.0]],
#
#                        [[2.0  , 2.0, 2.0],
#                         [2.0  , 2.0, 2.0 ],
#                         [2.0  , 2.0, 2.0]],
#
#                        # [[2.0  , 2.0, 2.0],
#                        #  [2.0  , 2.0, 2.0 ],
#                        #  [2.0  , 2.0, 2.0]]
#                        ]], requires_grad=True)
#
#     f = testsolver(a)
#     J1 = Jacobian(mtd="batchJacobian")
#     vG = tensorNewton(f,J1)
#     x = vG(x0)


    # Description: Function to test all the Jacobian options

    # x = torch.tensor([[[1.0,2.0],[3.0,0.6],[0.1,0.2]]],requires_grad=True)
    #
    # y0 = torch.zeros([1, 3, 2],requires_grad=True)
    # y = y0.clone() # has to do clone, it seems.
    # k0 = torch.tensor([2.0,0.5,0.6,1.0],requires_grad=True).repeat([3,1])
    # k = k0.clone()
    # y[:,:,0]   = k[:, 0] * x[:,:, 0] + k[:, 1] * x[:, :, 1] + x[:,:, 0] * x[:,:, 1]#k[:,0]*x[:,0]+k[:,1]*x[:,1]+x[:,0]*x[:,1]
    # y[:,:,1]  = k[:,2]*x[:,:,0]+k[:,3]*x[:,:,1]#k[:,2]*x[:,0]+k[:,3]*x[:,1]
    # # x = torch.rand((2,3,2), requires_grad=True)
    # # y = 2*x**3
    #
    # y = y.view(3,2)
    # jac = batchJacobian_AD(x,y)
    # print(jac.detach().numpy()) # expected:
    #tensor([[[4.0000, 1.5000],
    #     [0.6000, 1.0000]],
    #    [[2.6000, 3.5000],
    #     [0.6000, 1.0000]],
    #    [[2.2000, 0.6000],
    #     [0.6000, 1.0000]]], grad_fn=<CopySlices>)
    # xDict=dict(); yDict=dict()
    # xDict["u0"] = x
    # xDict["params"] = k
    # yDict["yP"] = y
    # J0 = Jacobian(mtd="batchJacobian_AD")
    # JAC, NAMES = getDictJac(xDict,yDict,J0,xfs=("params",'u0'),yfs=('yP',),ySel=([],),yPermuteDim=([],))
    #
    # f = nonlinearsolver()
    #
    # J1 = Jacobian(mtd="batchScalarJacobian_AD")
    # J2 = Jacobian(mtd="batchScalarJacobian_FD", func=f, settings={"dx":1e-2,"bounds":(0.0, 10.0)})
    # vG = tensorNewton(f,J1)
    # x0 = torch.tensor([1.0,1.,1.0,1.0, 1.0],requires_grad=True) # FD can be run in eval mode
    # x  = vG(x0)
    #
    # vG = tensorNewton(f,J2)
    # x0 = torch.tensor([1.0,1.,1.0,1.0, 1.0],requires_grad=True)
    # x  = vG(x0)
    #
    # vG = tensorNewton(f,J1,mtd="rtnobnd",lb=0.0,ub=1000.0,settings={"maxiter": 70, "ftol": 1e-6, "xtol": 1e-6, "alpha": 0.75})
    # x0 = torch.tensor([1.0,1.,1.0,1.0, 1.0],requires_grad=True)
    # x  = vG(x0)
    # print(x)
    #
    #
    #
    # # Testing with a simple case
    # x = torch.randn(3, 4, 5, requires_grad=True)  # Shape [nb1, nb2, nx]
    # y = x ** 2  # Simple quadratic relation
    #
    # # Calculate Jacobian
    # jacobian = batchJacobian_AD(x, y)
    # print("Jacobian:", jacobian)
    # print("Expected Gradient (2x):", 2 * x)

    # return jac
# testJacobian()
########################################################################################################################
