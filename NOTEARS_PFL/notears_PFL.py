import numpy as np
import torch
import torch.nn as nn
import utils

from NOTEARS_PFL.lbfgsb_scipy import LBFGSBScipy
from NOTEARS_PFL.trace_expm import trace_expm
import time


class LinearModel(nn.Module):
    def __init__(self, d):
        super(LinearModel, self).__init__()
        self.B = torch.nn.Parameter(torch.zeros(d, d))

    def forward(self, x):  # [n, d] -> [n, d]
        return x @ self.B

    def l1_reg(self):
        """Take l1 norm of linear weight"""
        reg = torch.norm(self.B, p=1)
        return reg

    def h_func(self):
        d = self.B.shape[0]
        h = trace_expm(self.B * self.B) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(d) + self.B * self.B / d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, d - 1)
        # h = (E.t() * M).sum() - d
        return h

    @torch.no_grad()
    def adj(self) -> np.ndarray:  # [d, d] -> [d, d]
        return self.B.cpu().detach().numpy()


class LinearModel_global(nn.Module):
    def __init__(self, k, d):
        super(LinearModel_global, self).__init__()
        self.B = torch.nn.Parameter(torch.zeros(k, d, d))

    def forward(self, x):  # [n, d] -> [n, d]
        return x @ self.B

    def l1_reg(self):
        """Take l1 norm of linear weight"""
        K, d = self.B.shape[0], self.B.shape[1]
        reg = 0
        for k in range(K):
            reg += torch.norm(self.B[k], p=1)
        return reg

    def h_func(self):
        K, d = self.B.shape[0], self.B.shape[1]
        h = np.zeros(K)
        G_h = np.zeros((K,d,d))
        for k in range(K):
            # h[k] = trace_expm(self.B[k] * self.B[k]) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
            M = torch.eye(d) + self.B[k] * self.B[k] / d  # (Yu et al. 2019)
            E = torch.matrix_power(M, d - 1)
            h[k] = (E.t() * M).sum() - d
            G_h[k, :, :] = E.t().detach().numpy() * self.B[k].detach().numpy() * 2
        return h, G_h

    @torch.no_grad()
    def adj(self, k) -> np.ndarray:  # [k, d, d] -> [k, d, d]
        B_adj = self.B.cpu().detach().numpy()
        return B_adj[k, :, :]

def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def solve_local_subproblem(cov_emp, local_model, global_model, beta, rho2):
    d = cov_emp.shape[0]
    sol = np.linalg.inv(cov_emp + rho2 * np.eye(d)) @ (rho2 * global_model - beta + cov_emp)
    local_model.B = torch.nn.Parameter(torch.from_numpy(sol))  # NOTE: updates model in-place


def solve_global_subproblem(local_models, global_model, lambda1, lambda2, alpha, betas, rho1, rho2):
    K = len(local_models)
    optimizer = LBFGSBScipy(global_model.parameters())
    betas_torch = torch.from_numpy(betas)

    def closure():
        optimizer.zero_grad()
        penalty, l1_reg, fro_reg, C = 0, 0, 0, 1
        h_val, G_h = global_model.h_func()
        l1_reg = lambda1 * global_model.l1_reg()
        for k in range(K):
            L_g = alpha[k] * G_h[k, :, :] + rho1 * h_val[k] * G_h[k, :, :] - betas_torch[k].detach().numpy() + rho2 * (global_model.B[k, :, :].detach().numpy() - local_models[k].B.detach().numpy())
            u_k = global_model.B[k, :, :].detach().numpy() - (1/C) * L_g
            penalty += 0.5 * torch.norm(global_model.B[k, :, :] - torch.from_numpy(u_k)) ** 2
        # fused graphical lasso
        # for k in range(K):
        #     for k2 in range(k+1, K):
        #         fro_reg += lambda2 * torch.norm((global_model.B[k, :, :] - global_model.B[k2, :, :]))
        # objective = penalty + l1_reg + fro_reg
        #The group graphical lasso
        for k in range(K):
            fro_reg += torch.norm(global_model.B[k, :, :]**2)
        objective = penalty + l1_reg + lambda2 * torch.sqrt(fro_reg)
        objective.backward()
        return objective
    optimizer.step(closure)  # NOTE: updates model in-place


def update_params(alpha, betas, rho1, rho2, local_models,
                       global_model, h, rho_max):
    K = len(local_models)
    for k in range(K):
        alpha[k] += rho1 * h[k]
        betas[k] += rho2 * (local_models[k].adj() - global_model.adj(k))
    if rho1 < rho_max:
        rho1 *= 1.5
    if rho2 < rho_max:
        rho2 *= 1.25
    return alpha, betas, rho1, rho2


def compute_consensus_distance(global_model_old, global_model):
    K = global_model_old.B.shape[0]
    consensus_distance = 0
    for k in range(K-1):
        consensus_distance += np.linalg.norm(global_model.adj(k) - global_model_old.adj(k),'fro') / np.linalg.norm(global_model.adj(k),2)
    # consensus_distance /= K
    return consensus_distance


def compute_empirical_covs(Xs):
    K, n, d = Xs.shape
    covs_emp = []
    for k in range(K):
        cov_emp = Xs[k].T @ Xs[k] / n / K
        covs_emp.append(cov_emp)
    return np.array(covs_emp)


def compute_empirical_covs2(Xs):
    n, d = Xs.shape
    cov_emp = Xs.T @ Xs / n
    return cov_emp


def notears_PFL(Xs: np.ndarray,
                        lambda1: float = 0.01,
                        lambda2: float = 0.01,
                        max_iter: int = 1000,
                        rho_max: float = 1e+16,
                        verbose: bool = False):
    """Solve ADMM problem using augmented Lagrangian.
    Args:
        Xs (np.ndarray): [K, n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        max_iter (int): max num of dual ascent steps
        rho_max (float): exit if rho1 >= rho_max and rho2 >= rho_max
        verbose (bool): Whether to print messages during optimization

    Returns:
        B_est (np.ndarray): [d, d] estimated weights
    """
    K, n, d = Xs.shape
    # Initialize local and global models
    local_models = [LinearModel(d) for _ in range(K)]
    global_model = LinearModel_global(K, d)
    # Initialize ADMM parameters
    rho1, rho2, h = 0.001, 0.001, np.inf
    alpha, betas = np.zeros(K), np.zeros((K, d, d))
    # Standardize data and compute empirical covariances
    Xs = Xs - np.mean(Xs, axis=1, keepdims=True)
    covs_emp = compute_empirical_covs(Xs)
    global_model_old = global_model
    # ADMM
    for t in range(1, max_iter + 1):
        # Solve local subproblem for each client
        for k in range(K):
            solve_local_subproblem(covs_emp[k], local_models[k], global_model.adj(k), betas[k], rho2)
        # Solve global subproblem
        solve_global_subproblem(local_models, global_model, lambda1, lambda2, alpha, betas, rho1, rho2)
        # Obtain useful value
        with torch.no_grad():
            h,_ = global_model.h_func()
        # Printing statements
        if verbose:
            consensus_distance = compute_consensus_distance(global_model_old, global_model)
            print("----- Iteration {} -----".format(t))
            print("rho1 {:.3E}, rho2 {:.3E}".format(rho1, rho2))
            print("h {:.3E}, consensus_distance {:.3E}".format(h.sum(), consensus_distance))
        # Update ADMM parameters
        alpha, betas, rho1, rho2 = update_params(alpha, betas, rho1, rho2, local_models,
                                                      global_model, h, rho_max)
        global_model_old = global_model
        # Terminate the optimization
        if rho1 >= rho_max and rho2 >= rho_max:
            break

    B_est = global_model.B.detach().cpu().numpy()
    return B_est


def notears_PFL_train(Xs: list,
                        lambda1: float = 0.01,
                        lambda2: float = 0.01,
                        max_iter: int = 1000,
                        rho_max: float = 1e+16,
                        verbose: bool = False):
    """Solve ADMM problem using augmented Lagrangian.
    Args:
        Xs (np.ndarray): [K, n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        max_iter (int): max num of dual ascent steps
        rho_max (float): exit if rho1 >= rho_max and rho2 >= rho_max
        verbose (bool): Whether to print messages during optimization

    Returns:
        B_est (np.ndarray): [d, d] estimated weights
    """
    K = len(Xs)
    n,d = Xs[0].shape
    # Initialize local and global models
    local_models = [LinearModel(d) for _ in range(K)]
    global_model = LinearModel_global(K, d)
    # Initialize ADMM parameters
    rho1, rho2, h = 0.01, 0.01, np.inf
    alpha, betas = np.zeros(K), np.zeros((K, d, d))
    # Standardize data and compute empirical covariances
    covs_emp_site=list()
    for ii in range(K):
        Xss = Xs[ii] - np.mean(Xs[ii], axis=1, keepdims=True)
        covs_emp = compute_empirical_covs2(Xss)
        covs_emp_site.append(covs_emp)

    global_model_old = global_model
    # ADMM
    for t in range(1, max_iter + 1):
        # Solve local subproblem for each client
        for k in range(K):
                solve_local_subproblem(covs_emp_site[k], local_models[k], global_model.adj(k), betas[k], rho2)
        # Solve global subproblem
        solve_global_subproblem(local_models, global_model, lambda1, lambda2, alpha, betas, rho1, rho2)
        # Obtain useful value
        with torch.no_grad():
            h,_ = global_model.h_func()
        # Printing statements
        if verbose:
            consensus_distance = compute_consensus_distance(global_model_old, global_model)
            print("----- Iteration {} -----".format(t))
            print("rho1 {:.3E}, rho2 {:.3E}".format(rho1, rho2))
            print("h {:.3E}, consensus_distance {:.3E}".format(h.sum(), consensus_distance))
        # Update ADMM parameters
        alpha, betas, rho1, rho2 = update_params(alpha, betas, rho1, rho2, local_models,
                                                      global_model, h, rho_max)
        global_model_old = global_model
        # Terminate the optimization
        if rho1 >= rho_max and rho2 >= rho_max:
            break

    B_est = global_model.B.detach().cpu().numpy()
    return B_est


def main_variable():
    from NOTEARS_PFL import utils
    from NOTEARS_PFL.postprocess import postprocess

    # Configuration of torch
    torch.set_default_dtype(torch.double)

    # Generate data
    utils.set_random_seed(1)
    K ,rho= 10, 0.1
    graph_type, sem_type = 'ER', 'gauss'
    # d = [10,20,30,40,50,60,70,80]
    d = [80]
    for dd in d:
        s0 = dd
        n = int(0.3*dd)
        B_bin_true = utils.simulate_dags_PFL(dd, s0, K, rho, graph_type)
        B_true = []
        Xs = np.zeros((K, n, dd))
        for k in range(K):
            W_true_k = utils.simulate_parameter(B_bin_true[k])
            B_true.append(W_true_k)
        # initial x
            Xs[k, :, :] = utils.simulate_linear_sem(W_true_k, n, sem_type)

        # Run NOTEARS-PFL
        # utils.set_random_seed(1)
        B_est = notears_PFL(Xs, lambda1=0.01, lambda2=0.01, verbose=False)
        error_mean, precision_mean, Fscore_mean, shd_mean = 0, 0, 0, 0
        for k in range(K):
            B_processed = postprocess(B_est[k], threshold=0.3)
            acc = utils.count_accuracy(B_bin_true[k], B_processed!=0)
            # print(acc)
            error_mean += acc['Error']
            precision_mean += acc['precision']
            Fscore_mean += acc['F_score']
            shd_mean += acc['shd']
        print('paramter {}: Error mean: {}, precision mean: {}, F-score mean: {}, shd mean: {}'.format(dd,
                                                                                                       error_mean / K,
                                                                                                       precision_mean / K,
                                                                                                       Fscore_mean / K,
                                                                                                       shd_mean / K))


if __name__ == '__main__':
    time_start = time.time()
    main_variable()
    time_end = time.time()
    time_elapsed = time_end - time_start
    print('time elapsed: {}'.format(time_elapsed))
     # main_site()
     # main_perturbation()
