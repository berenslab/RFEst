class splineLNP(splineBase):

    def __init__(self, X, y, dims, df, smooth='cr', compute_mle=True, **kwargs):
        
        super().__init__(X, y, dims, df, smooth, compute_mle, **kwargs)

    def cost(self, b):

        """
        Negetive Log Likelihood.
        """
        
        XS = self.XS
        y = self.y
        dt = self.dt
        
        def nonlin(x):
            return np.log(1 + np.exp(x)) + 1e-17


        filter_output = nonlin(XS @ b).flatten()
        r = dt * filter_output

        term0 = - np.log(r) @ y # spike term from poisson log-likelihood
        term1 = np.sum(r) # non-spike term

        neglogli = term0 + term1
        
        if self.lambd:
            l1 = np.sum(np.abs(b))
            l2 = np.sqrt(np.sum(b**2)) 
            neglogli += self.lambd * ((1 - self.alpha) * l2 + self.alpha * l1)

        return neglogli

    def fit(self, p0=None, num_iters=5, alpha=0.5, lambd=0.05, gamma=0.0,
            step_size=1e-2, tolerance=10, verbal=1, random_seed=2046):

        self.lambd = lambd # elastic net parameter - global weight
        self.alpha = alpha # elastic net parameter (1=L1, 0=L2)
        self.gamma = gamma # nuclear norm parameter
        
        self.n_subunits = num_subunits
        self.num_iters = num_iters   
        
        if p0 is None:
            p0 = self.b_spl
        
        self.b_opt = self.optimize_params(p0, num_iters, step_size, tolerance, verbal)   
        self.w_opt = self.S @ self.b_opt.reshape(self.n_b, self.n_subunits)
