# external imports
import numpy as np
from scipy.linalg import expm, cholesky

# local imports
from utils import get_logger, get_t_span, TIME_STEPS, EPS
from integrators import IntegratorType, integrate_rk4, integrate_euler_maruyama, \
    integrate_analytic_ode, integrate_analytic_sde


class Plant:

    # TODO: add function to compute time constant and delay of the equivalent FOPDT (reduced order) plant model,
    # using Skogestad's half rule. This can be used in the PID tuning rules.

    def __init__(self, dims: int, x_0: np.ndarray = None, u_0: np.ndarray = None,
                 A: np.ndarray = None, B: np.ndarray = None, C: np.ndarray = None, D: np.ndarray = None,
                 Q: np.ndarray = None, R: np.ndarray = None):
        '''
        Initialise a linear plant, representing a dynamical system. The state of the plant
        is a vector of dimension `dims`. The plant accepts a scalar control input and produces
        a scalar measurement output. The plant is affected by process noise and measurement noise.
        
        When in SDE (stochastic differential equation) mode (the default setting in the Simulator),
        the equations describing the plant are:

        - dx = (A x + B u) dt + G dW_t
        - y = C x + D u
        - y_meas = y + w_meas

        where:

        - `u`: the control input to the plant. Shape: (1, 1) (scalar)
        - `y_meas`: the measurement output from the plant. Shape: (1, 1) (scalar)
        - `x`: the state of the plant. Shape (`dims`, 1) (column vector)
        - `dW_t`: Wiener increments. Shape: (`dims`, 1) (column vector)
        - `w_meas`: measurement noise, drawn from a Gaussian distribution. Shape: (1, 1) (scalar)

        The state-space matrices A, B, C, D are set using the `set_state_space_matrices` method.
        The noise matrices Q, R are set using the `set_noise_matrices` method. Here, Q is
        the diffusion matrix for the stochastic process, such that G @ G.T = Q, and R is the 
        1x1 covariance matrix (i.e. variance) for the measurement noise samples w_meas, i.e.
        w_meas ~ N(0, R).

        When in ODE (ordinary differential equation) mode (`use_ode_mode = True`), 
        the equations describing the plant are:

        - dx/dt = A x + B u + w_proc
        - y = C x + D u
        - y_meas = y + w_meas

        where:

        - `u`: the control input to the plant. Shape: (1, 1) (scalar)
        - `y_meas`: the measurement output from the plant. Shape: (1, 1) (scalar)
        - `x`: the state of the plant. Shape (`dims`, 1) (column vector)
        - `w_proc`: process noise, drawn from a Gaussian distribution. Shape: (`dims`, 1) (column vector)
        - `w_meas`: measurement noise, drawn from a Gaussian distribution. Shape: (1, 1) (scalar)

        Here, both Q and R are noise covariance matrices for their noise terms i.e.
        w_proc ~ N(0, Q) and w_meas ~ N(0, R).

        NOTE: the SDE and ODE plant models are not mathematically equivalent, since the meaning
        of Q differs between them.

        ### Arguments
        #### Required
        - `dims` (int): the dimension of the plant state vector.
        #### Optional
        - `x_0` (np.ndarray, default = None): initial state of the plant. Shape: (`dims`, 1) (column vector)
        - `u_0` (np.ndarray, default = None): initial control input to the plant. Shape: (1, 1) (scalar)
        - `A` (np.ndarray, default = None): state transition matrix. Shape: (`dims`, `dims`) (square matrix)
        - `B` (np.ndarray, default = None): control input matrix. Shape: (`dims`, 1) (column vector)
        - `C` (np.ndarray, default = None): measurement matrix. Shape: (1, `dims`) (row vector)
        - `D` (np.ndarray, default = None): feedthrough matrix. Shape: (1, 1) (scalar)
        - `Q` (np.ndarray, default = None): process noise diffusion matrix 
        (or if in ODE mode, it is the covariance matrix). Shape: (`dims`, `dims`) (positive semi-definite matrix)
        - `R` (np.ndarray, default = None): measurement noise covariance matrix. Shape: (1, 1) (non-negative scalar)
        '''

        self.logger = get_logger()

        if isinstance(dims, int) and dims > 0:
            self.dims = dims
        else:
            raise ValueError('Plant state dimension must be a positive integer.')
        
        # set initial values (constant)
        self.u_0 = u_0 if u_0 is not None else np.array([[0.0]])  # shape (1, 1)
        self.x_0 = x_0 if x_0 is not None else np.zeros((self.dims, 1))  # shape (dims, 1)

        # set values (these get updated by the control system as the simulation runs)
        self.u = self.u_0
        self.x = self.x_0

        # get time steps
        # TODO: these need to be overwritten by the Simulator if we allow the user to change the time steps in the GUI
        self.dt_int = TIME_STEPS['DT_INT']
        self.dt_anim = TIME_STEPS['DT_ANIM']

        # set state-space matrices and noise matrices if provided
        self.set_all_arrays(A, B, C, D, Q, R)

    def set_state_space_matrices(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray):
        '''
        The dynamics of the plant are modelled as a linear time-invariant system in state-space form.
        This function sets the attributes `self.A`, `self.B`, `self.C`, and `self.D`.

        - `A`: state transition matrix. Shape: (dims, dims) (square matrix)
        - `B`: control input matrix. Shape: (dims, 1) (column vector)
        - `C`: measurement matrix. Shape: (1, dims) (row vector)
        - `D`: feedthrough matrix. Shape: (1, 1) (scalar)
        '''

        # check matrices are all correct sizes
        if A.shape != (self.dims, self.dims) or B.shape != (self.dims, 1) or C.shape != (1, self.dims) or D.shape != (1, 1):
            raise ValueError('State-space matrices have incorrect dimensions.')
        
        # check eigenvalues of A have negative real part (for stability)
        eigs = np.linalg.eigvals(A)
        re_eigs = np.real(eigs)
        if np.max(re_eigs) > 0 or np.sum(np.isclose(eigs, 0)) >= 2:
            self.logger.warning(f'The plant is open-loop unstable. Eigenvalues of A: {eigs}.')
        elif np.isclose(np.max(re_eigs), 0):
            self.logger.warning(f'The plant is marginally stable (not asymptotically stable). Eigenvalues of A: {eigs}.')
        else:
            # check that fastest plant time constant is not too small compared to sampling period (avoid numerical issues)
            t_sample = TIME_STEPS['DT_ANIM']
            min_time_constant = np.min(-1 / re_eigs)
            if min_time_constant < 5 * t_sample:
                self.logger.warning(f'Plant is asymptotically stable, but has fast dynamics compared to sampling period. '
                    f'Numerical issues may arise. Eigenvalues of A: {eigs}. Sampling period: {t_sample}. '
                    f'Fastest time constant: {min_time_constant}.')
                
        # set cached matrices and values - used in plant integration and controller design
        # t_span_0: fixed time points for one integration interval, starting from zero
        self.t_span_0 = get_t_span(0, self.dt_anim, self.dt_int)
        # exp_A_t_span: precompute the matrix exponential of A * t for all time points in t_span_0
        A_t_span = np.outer(self.t_span_0, A.flatten()).reshape(self.t_span_0.shape[0], A.shape[0], A.shape[1])  # shape (num_steps, dims, dims)
        self.exp_A_t_span = expm(A_t_span)  # shape (num_steps, dims, dims)
        # A_inv_exp_At_minus_I: precompute A^{-1} @ (e^{A t} - I)
        self.A_inv_exp_At_minus_I = np.linalg.solve(A, (self.exp_A_t_span - np.eye(A.shape[0])))  # shape (num_steps, dims, dims)
        # G_0: the transfer function G(s) = C @ (sI - A)^{-1} @ B + D evaluated at s = 0
        self.G_0 = D - C @ np.linalg.solve(A, B)

        # set state space matrices
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def set_noise_matrices(self, Q: np.ndarray = None, R: np.ndarray = None):
        '''
        Set the process/measurement noise matrices `self.Q` and `self.R`.

        - `Q`: process noise matrix. In SDE mode (`use_ode_mode = False`) this is the diffusion
        matrix; in ODE mode (`use_ode_mode = True`) this is the covariance matrix of `w_proc`.
        Shape: (dims, dims) (positive semi-definite matrix)
        - `R`: measurement noise covariance matrix. Shape: (1, 1) (non-negative scalar)
        '''
        if Q is None:
            Q = np.zeros((self.dims, self.dims))
        if R is None:
            R = np.array([[0.0]])

        # check Q and R are acceptable (square, symmetric and positive semidefinite)
        if Q.shape != (self.dims, self.dims) or R.shape != (1, 1):
            raise ValueError('Noise matrices Q and R have incorrect dimensions.')
        elif not np.allclose(Q, Q.T):
            raise ValueError('Process noise matrix Q must be symmetric.')
        elif not np.all(np.linalg.eigvals(Q) >= 0) or R[0, 0] < 0:
            raise ValueError('Noise matrices Q and R must be positive semidefinite.')

        # set noise matrices
        self.Q = Q
        self.R = R

    def set_cached_arrays(self):
        '''
        Set cached arrays and matrices that are used in plant integration and controller design.
        This is to be called every time the plant matrices are changed.

        **Attributes set:**
        - `t_span_0`: fixed time points for one integration interval, starting from zero. Shape: (num_steps,)
        - `exp_A_t_span`: matrix exponential of A * t for all times in t_span_0. Shape: (num_steps, dims, dims)
        - `A_inv_exp_At_minus_I`: A^{-1} @ (e^{A t} - I) for all times in t_span_0. Shape: (num_steps, dims, dims)
        - `G_0`: the transfer function G(s) = C @ (sI - A)^{-1} @ B + D evaluated at s = 0. Shape: (1, 1) (scalar)
        - `Phi`: state transition matrix for the integrated dynamics from 0 to dt_int. Shape: (dims, dims)
        - `Gamma`: multiplier for u in discretised SDE state update equation. Shape: (dims, 1)
        - `Q_d`: integrated process noise covariance matrix for discretised SDE model. Shape: (dims, dims)
        - `Phi_last`: Phi for the last time step of t_span_0. Shape: (dims, dims)
        - `Gamma_last`: Gamma for the last time step of t_span_0. Shape: (dims, 1)
        - `Q_d_last`: Q_d for the last time step of t_span_0. Shape: (dims, dims)
        - `G_cov`: covariance matrix of the Wiener increments in the SDE model, such that G_cov @ G_cov.T = Q. Shape: (dims, dims)
        '''

        # get current state space matrices
        A, B, C, D, Q, R, dims = self.A, self.B, self.C, self.D, self.Q, self.R, self.dims

        # compute ODE discretisation matrices
        self.t_span_0 = get_t_span(0, self.dt_anim, self.dt_int)
        self.num_steps = self.t_span_0.shape[0]
        A_t_span = np.outer(self.t_span_0, A.flatten()).reshape(self.num_steps, dims, dims)
        self.exp_A_t_span = expm(A_t_span)
        self.A_inv_exp_At_minus_I = np.linalg.solve(A, (self.exp_A_t_span - np.eye(dims)))
        self.G_0 = D - C @ np.linalg.solve(A, B)

        # M: Van Loan block matrix
        # source: https://personales.upv.es/asala/DocenciaOnline/material/DiscretizRuidoVanLoanEN.pdf
        M = np.block([[A,                  Q   ],
                      [np.zeros_like(A),   -A.T]])  # shape (2 * dims, 2 * dims)
        
        # compute SDE discretisation matrices Phi, Gamma and Q_d based on a time step of self.dt_int
        exp_M = expm(M * self.dt_int)  # shape (2 * dims, 2 * dims)
        self.Phi = exp_M[:dims, :dims]
        self.Q_d = exp_M[:dims, dims:] @ self.Phi.T
        # check Q_d is close to being symmetric (it should already be)
        if not np.allclose(self.Q_d, self.Q_d.T, atol=EPS):
            self.logger.warning(f'''Integrated process noise covariance matrix Q_d is 
                not symmetric. Q_d: {self.Q_d}. Taking the symmetric component and continuing.''')
        # take the symmetric component (ideally no change to within numerical precision)
        self.Q_d = 0.5 * (self.Q_d + self.Q_d.T)
        self.Gamma = np.linalg.solve(A, (self.Phi - np.eye(dims)) @ B)  # shape (dims, 1)

        # compute SDE discretisation matrices Phi, Gamma and Q_d based on the last time step of self.t_span_0 (to check for any discrepancies with the above)
        dt_int_last = self.dt_anim - self.t_span_0[-2]
        if np.isclose(dt_int_last, self.dt_int, atol=EPS):
            # last time step is equal to self.dt_int, so no need to recompute
            self.Phi_last = self.Phi
            self.Gamma_last = self.Gamma
            self.Q_d_last = self.Q_d
        else:
            exp_M_last = expm(M * dt_int_last)  # shape (2 * dims, 2 * dims)
            self.Phi_last = exp_M_last[:dims, :dims]
            self.Q_d_last = exp_M_last[:dims, dims:] @ self.Phi_last.T
            # check Q_d_last is close to being symmetric (it should already be)
            if not np.allclose(self.Q_d_last, self.Q_d_last.T, atol=EPS):
                self.logger.warning(f'''Integrated process noise covariance matrix for the last time step Q_d_last is 
                    not symmetric. Q_d_last: {self.Q_d_last}. Taking the symmetric component and continuing.''')
            # take the symmetric component (ideally no change to within numerical precision)
            self.Q_d_last = 0.5 * (self.Q_d_last + self.Q_d_last.T)
            self.Gamma_last = np.linalg.solve(A, (self.Phi_last - np.eye(dims)) @ B)  # shape (dims, 1)

        # NOTE: G_cov is currently unused
        try:
            self.G_cov = cholesky(Q, lower=True)  # shape (dims, dims)
        except np.linalg.LinAlgError:  # Q = 0 gives error due to zero eigenvalues, so set to zero manually
            self.G_cov = np.zeros((dims, dims))

    def set_all_arrays(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, \
            Q: np.ndarray, R: np.ndarray):
        '''
        Utility function to call 1) `set_state_space_matrices`, 2) `set_noise_matrices` and 
        3) `set_cached_arrays`, to update all plant matrices and cached arrays at once.
        '''
        self.set_state_space_matrices(A, B, C, D)
        self.set_noise_matrices(Q, R)
        self.set_cached_arrays()

    def sample_process_noise(self, n: int = 1) -> np.ndarray:
        '''
        Returns sample(s) of process noise, drawn from its Gaussian distributions.
        
        ### Arguments
        #### Optional
        - `n` (int, default = 1): number of noise samples to generate.
        
        ### Returns
        - `np.ndarray`: process noise samples. Shape (dims, `n`) (array of column vectors)
        '''        

        w_proc = np.random.multivariate_normal(mean=np.zeros(self.dims), cov=self.Q, size=n, 
            check_valid='raise').T  # shape (dims, n)
        
        return w_proc

    def integrate_dynamics(self, t_start: float, t_stop: float, dt: float, 
            method: IntegratorType | None = None, use_ode_mode: bool = False) \
                -> tuple[np.ndarray, np.ndarray]:
        '''
        Integrates the equations describing the plant system from `t_start` to `t_stop` inclusive, using
        a step size of `dt`. The initial state is `self.x` at time `t_start`.
        The control input is `self.u` at all time steps within the integration period.
        At the end, the state at `t_stop` is stored in `self.x`.
        The measured output is not computed here.

        Integration methods:
        - ODE mode (`use_ode_mode = True`): `IntegratorType.RK4`, `IntegratorType.ANALYTIC_ODE`
        - SDE mode (`use_ode_mode = False`): `IntegratorType.EULER_MARUYAMA`, `IntegratorType.ANALYTIC_SDE`
       
        ### Arguments
        #### Required
        - `t_start` (float): start time of integration
        - `t_stop` (float): stop time of integration
        - `dt` (float): step size for numerical integration
        #### Optional
        - `method` (IntegratorType | None, default = None): integration method.
        If None, a mode-appropriate default is used (`RK4` for ODE mode, `EULER_MARUYAMA` for SDE mode).
        - `use_ode_mode` (bool, default = False): if True, run the ODE model where process noise
        is sampled once per integration interval and treated as a constant drift term. If False,
        run the SDE model where process noise is treated as a stochastic diffusion term.
        
        ### Returns
        - `tuple[np.ndarray, np.ndarray]`: arrays of time points and state trajectory (including both endpoints).
        '''

        # set the default integration method to the numerical method if not provided
        if method is None:
            method = IntegratorType.RK4 if use_ode_mode else IntegratorType.EULER_MARUYAMA 

        if use_ode_mode:
            # we are modelling the plant using an ODE, so noise is treated as a constant 
            # disturbance across the integration interval
            match method:
                case IntegratorType.RK4:
                    t_span, x_span = integrate_rk4(self, t_start, t_stop, dt)
                case IntegratorType.ANALYTIC_ODE:
                    t_span, x_span = integrate_analytic_ode(self, t_start)
                case _:
                    raise ValueError(
                        f'Invalid integration method for use_ode_mode = True: {method}. '
                        'Expected one of: IntegratorType.RK4, IntegratorType.ANALYTIC_ODE.')
        else:
            # we are modelling the plant using an SDE, so noise is treated as a stochastic 
            # diffusion term across the integration interval
            match method:
                case IntegratorType.EULER_MARUYAMA:
                    t_span, x_span = integrate_euler_maruyama(self, t_start, t_stop, dt)
                case IntegratorType.ANALYTIC_SDE:
                    t_span, x_span = integrate_analytic_sde(self, t_start)
                case _:
                    raise ValueError(
                        f'Invalid integration method for use_ode_mode = False: {method}. '
                        'Expected one of: IntegratorType.EULER_MARUYAMA, IntegratorType.ANALYTIC_SDE.')

        # store final state in plant attributes
        # this becomes the initial state for the next time span when `integrate_dynamics` is called again
        self.x = x_span[:, -1].reshape(self.dims, 1)
        return t_span, x_span
        
    def calc_y(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
        Calculate the true output y (no measurement noise is added)

        ### Arguments
        - `x` (np.ndarray): state vectors. Shape: (dims, n) (array of column vectors)
        - `u` (np.ndarray): control inputs. Shape: (1, n) (row vector)

        ### Returns
        - `np.ndarray`: output vectors. Shape: (1, n) (row vector)
        '''
        return self.C @ x + self.D @ u
    
    def sample_measurement_noise(self, n: int = 1) -> np.ndarray:
        '''
        Returns sample(s) of measurement noise, drawn from its Gaussian distribution.
        
        ### Arguments
        #### Optional
        - `n` (int, default = 1): number of noise samples to generate.
        
        ### Returns
        - `np.ndarray`: measurement noise samples. Shape (1, `n`) (row vector)
        '''
        std_dev = np.sqrt(self.R[0, 0])
        return np.random.normal(loc=0.0, scale=std_dev, size=(1, n))  # shape (1, n)
    
    def sample_measurement(self) -> np.ndarray:
        '''
        Sample of the noisy measurement y, given the current state `x` and control input `u`, 
        and including measurement noise.

        This function is only called once, at the start of the first frame.
        
        Set the attributes `self.y` (the true output without noise) and 
        `self.y_meas` (the measured output with noise).

        ### Returns
        - `np.ndarray`: the measured output with noise. Shape: (1, 1) (scalar)
        '''
        self.y = self.calc_y(self.x, self.u)  # shape (1, 1)
        self.y_meas = self.y + self.sample_measurement_noise(n=1)  # shape (1, 1)
        return self.y_meas
