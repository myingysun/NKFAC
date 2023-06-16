import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pdb


class Ada_NKFAC(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.1,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 amsgrad=False,
                 stat_decay=0.95,
                 damping1=0.01,
                 damping2=0.01,
                 weight_decay=0,
                 single_gpu=False,
                 TCov=100,
                 TInv=1000,
                 known_modules={'Linear', 'Conv2d'},
                 K = 1):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,amsgrad=amsgrad,
                        weight_decay=weight_decay)
        super(Ada_NKFAC, self).__init__(model.parameters(), defaults)
        self.XXHandler = ComputeXX()
        self.GGHandler = ComputeGG()
        self.known_modules = known_modules
        self.modules = []
        self.model = model
        self.damping2 = damping2
        self.damping1 = damping1
        self.m_xx = {}
        self.D_xx = {}
        self.m_gg = {}
        self.D_gg = {}

        self.stat_decay = stat_decay
        self.TCov = TCov
        self.TInv = TInv
        self.steps = 0
        self._prepare_model()
        self.single_gpu = single_gpu
        self.K = K
        if not self.single_gpu:
            import torch.distributed as dist
            self.size = dist.get_world_size()


    def _save_input(self, module, input):
        if self.steps % self.TCov == 0 and self.model.training is True:
            xx = self.XXHandler(input[0].data, module)
            if self.steps == 0:
                self.m_xx[module] = torch.diag(xx.new(xx.size(0)).fill_(0))
            self.m_xx[module] += (xx/(torch.norm(xx)+1e-12) - self.m_xx[module]) * (1 - self.stat_decay)


    def _save_grad_output(self, module, grad_input, grad_output):
        if self.steps % self.TCov == 0 and self.model.training is True:
            gg = self.GGHandler(grad_output[0].data, module)
            if self.steps == 0:
                self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(0))
            self.m_gg[module] += (gg /(torch.norm(gg)+1e-12) - self.m_gg[module]) * (1 - self.stat_decay)


    def _prepare_model(self):
            for module in self.model.modules():
                classname = module.__class__.__name__
                if classname in self.known_modules:
                    self.modules.append(module)
                    module.register_forward_pre_hook(self._save_input)
                    module.register_backward_hook(self._save_grad_output)


    def _update_inv(self, m):
        if self.steps <= 2*self.TInv or self.steps % (500*self.TInv) == 0:
            max_ev_xx = max_eignvalue(self.m_xx[m])
            damping1 = max(1e-4, max_ev_xx * self.damping1)
            max_ev_gg = max_eignvalue(self.m_gg[m])
            damping2 = max(1e-4, max_ev_gg * self.damping2)
            self.D_xx[m] = torch.linalg.inv(
                self.m_xx[m] + damping1 * torch.eye(self.m_xx[m].size(0), dtype=self.m_xx[m].dtype,
                                                    device=self.m_xx[m].device))
            self.D_gg[m] = torch.linalg.inv(
                self.m_gg[m] + damping2 * torch.eye(self.m_gg[m].size(0), dtype=self.m_gg[m].dtype,
                                                    device=self.m_gg[m].device))
        else:
            max_ev_xx = max_eignvalue(self.m_xx[m])
            damping1 = max(1e-4, max_ev_xx * self.damping1)
            max_ev_gg = max_eignvalue(self.m_gg[m])
            damping2 = max(1e-4, max_ev_gg * self.damping2)
            self.D_xx[m] = newton_inv(
                self.m_xx[m] + damping1 * torch.eye(self.m_xx[m].size(0), dtype=self.m_xx[m].dtype,
                                                    device=self.m_xx[m].device), self.D_xx[m], self.K)
            self.D_gg[m] = newton_inv(
                self.m_gg[m] + damping2 * torch.eye(self.m_gg[m].size(0), dtype=self.m_gg[m].dtype,
                                                    device=self.m_gg[m].device), self.D_gg[m], self.K)

    @staticmethod
    def _get_matrix_form_grad(m, classname):
        if classname == 'Conv2d':
            p_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1)  # n_filters * (in_c * kw * kh)
        else:
            p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat


    def _get_modified_grad(self, m, p_grad_mat):
        v = self.D_gg[m] @ p_grad_mat @ self.D_xx[m]
        v = v * (p_grad_mat.norm() / (v.norm() + 1e-12))
        if m.bias is not None:
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]
        return v


    def _update_grad(self, update_inv=True):
        for m in self.modules:
            classname = m.__class__.__name__
            if m.weight.grad is not None:
                if self.steps % self.TInv == 0 and update_inv is True:
                    self._update_inv(m)

                p_grad_mat = self._get_matrix_form_grad(m, classname)
                v = self._get_modified_grad(m, p_grad_mat)
                m.weight.grad.data.copy_(v[0])
                if m.bias is not None:
                    m.bias.grad.data.copy_(v[1])


    def allreduce_factors(self):
        if self.size == 1:
            return
        for m in self.model.modules():
            classname = m.__class__.__name__
            if classname in self.known_modules and m.weight.grad is not None:
                dist.all_reduce(self.m_xx[m])
                self.m_xx[m] /= self.size
                dist.all_reduce(self.m_gg[m])
                self.m_gg[m] /= self.size

    @torch.no_grad()
    def _step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                amsgrad = group['amsgrad']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1  # *(1-self.stat_decay**(int(self.steps/self.TCov)+1))
                # GC operation and stepweight decay
                G_grad = (exp_avg / denom).add(p.data, alpha=group['weight_decay'])
                # G_grad=(exp_avg/denom)
                p.data.add_(G_grad, alpha=-step_size)
        return loss

    @torch.no_grad()
    def step(self, closure=None):
        if self.steps % self.TInv == 0 and not self.single_gpu:
            self.allreduce_factors()
        self._update_grad()
        self._step(closure)
        self.steps += 1


#=================================================================================
class ComputeXX:

    @classmethod
    def compute_XX(cls, x, layer):
        return cls.__call__(x, layer)

    @classmethod
    def __call__(cls, x, layer):
        with torch.no_grad():
            if isinstance(layer, nn.Linear):
                xx = cls.linear(x, layer)
            elif isinstance(layer, nn.Conv2d):
                xx = cls.conv2d(x, layer)
            else:
                xx = None
        return xx

    @staticmethod
    def conv2d(x, layer):
        with torch.no_grad():
            x = _extract_patches(x, layer.kernel_size, layer.stride, layer.padding)
            if layer.bias is not None:
                x = torch.cat([x, x.new(x.size(0), 1).fill_(1)], 1)
            #x= x-x.mean(dim=0,keepdim=True)
            if x.size(0) != 0:
                xx=x.t() @ (x *(1/ (x.size(0))))
            else:
                xx = x.t() @ x
        return xx


    @staticmethod
    def linear(x, layer):
        with torch.no_grad():
            if x.dim()>2:
                   x=x.view(-1,x.size(-1))
            if layer.bias is not None:
                x = torch.cat([x, x.new(x.size(0), 1).fill_(1)], 1)
            #x= x-x.mean(dim=0,keepdim=True)
            if x.size(0) != 0:
                xx = x.t() @ (x * (1 / (x.size(0))))
            else:
                xx = x.t() @ x
        return xx



class ComputeGG:

    @classmethod
    def compute_gg(cls, g, layer):
        return cls.__call__(g, layer)

    @classmethod
    def __call__(cls, g, layer):
        with torch.no_grad():
            if isinstance(layer, nn.Conv2d):
                gg = cls.conv2d(g, layer)
            elif isinstance(layer, nn.Linear):
                gg = cls.linear(g, layer)
            else:
                gg = None
        return gg

    @staticmethod
    def conv2d(g, layer):
        with torch.no_grad():
            # g: batch_size * n_filters * out_h * out_w
            # n_filters is actually the output dimension (analogous to Linear layer)
            #print(g.size())
            #g = g.transpose(1, 2).transpose(2, 3)
            g=g.permute(0,2,3,1)
            g = try_contiguous(g)
            #print(g.size()) *(1/ g.size(0))
            g = g.view(-1, g.size(-1))
            gg = g.t() @ g
        return gg

    @staticmethod
    def linear(g, layer):
        with torch.no_grad():
            # g: batch_size * out_dim *(1/ g.size(0))
            if g.dim()>2:
                   g=g.view(-1,g.size(-1))
            gg = g.t() @ g
        return gg



def try_contiguous(x):
    with torch.no_grad():
        if not x.is_contiguous():
            x = x.contiguous()
    return x



def _extract_patches(x, kernel_size, stride, padding):
    with torch.no_grad():
        if padding[0] + padding[1] > 0:
            x = F.pad(x, (padding[1], padding[1], padding[0],padding[0])).data  # Actually check dims
        x = F.unfold(x,kernel_size=kernel_size,padding=0, stride=stride).permute(0,2,1)
        x=x.contiguous().view(x.size(0)*x.size(1),x.size(2))
    return x


def max_eignvalue(A, numIters=10):
    with torch.no_grad():
        v = torch.ones(A.size(0),1, dtype=A.dtype, device=A.device)
        for i in range(numIters):
            u=(A*v).sum(dim=0,keepdim=True)
            u=u*(1/u.norm())
            v=(A*u).sum(dim=1,keepdim=True)
            max_ev=v.norm()
    return max_ev



def newton_inv(A, A_inv, K=1):
    with torch.no_grad():
        tmp_I = torch.eye(A.size(0), dtype=A.dtype, device=A.device)
        for k in range (K):
            tmp = A_inv @ A
            iter_norm = torch.norm(tmp_I - tmp)
            if (iter_norm >= 1).all():
                eta = 1 / torch.norm(tmp)
                A_inv = ((1 + eta)*tmp_I - eta * tmp) @ A_inv
            else:
                A_inv = (2*tmp_I - tmp) @ A_inv
    return A_inv


