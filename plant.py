# external imports
import numpy as np
from scipy.linalg import expm

# local imports
from utils import get_logger, get_t_span, TIME_STEPS


class Plant:

    # TODO: add function to compute time constant and delay of the equivalent FOPDT (reduced order) plant model,
    # using Skogestad's half rule. This can be used in the PID tuning rules.

    def __init__(self, dims: int, x_0: np.ndarray = None, u_0: np.ndarray = None,
                 A: np.ndarray = None, B: np.ndarray = None, C: np.ndarray = None, D: np.ndarray = None,
                 Q: np.ndarray = None, R: np.ndarray = None):
        '''
        Initialise a linear plant, representing a dynamical system. The state of the plant
        is a vector of dimension `dims`. The plant accepts a scalar control input and produces
        a scalar measurement output. The plant is affected by process noise and measurement noise, 
        modelled by the continuous-time equations:

        - dx/dt = A x + B u + w_proc
        - y = C x + D u
        - y_meas = y + w_meas

        where:

        - `u`: the control input to the plant. Shape: (1, 1) (scalar)
        - `y_meas`: the measurement output from the plant. Shape: (1, 1) (scalar)
        - `x`: the state of the plant. Shape (`dims`, 1) (column vector)

        The state-space matrices A, B, C, D are set using the `set_state_space_matrices` method.
        The noise covariance matrices Q, R are set using the `set_noise_covariances` method.

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
        - `Q` (np.ndarray, default = None): process noise covariance matrix. Shape: (`dims`, `dims`) (positive semi-definite matrix)
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

        # set state-space matrices and noise covariances if provided
        self.set_state_space_matrices(A, B, C, D)
        self.set_noise_covariances(Q, R)

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
                
        # set cached matrices
        self.t_span_0 = get_t_span(0, self.dt_anim, self.dt_int)
        A_t_span = np.outer(self.t_span_0, A.flatten()).reshape(self.t_span_0.shape[0], A.shape[0], A.shape[1])  # shape (num_steps, dims, dims)
        self.exp_A_t_span = expm(A_t_span)  # shape (num_steps, dims, dims)
        self.A_inv = np.linalg.inv(A)  # shape (dims, dims)

        # set state space matrices
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def set_noise_covariances(self, Q: np.ndarray = None, R: np.ndarray = None):
        '''
        The disturbances to the plant are modelled as zero-mean Gaussian noise with 
        covariance matrices Q and R. This function sets the attributes `self.Q` and `self.R`.

        - `Q`: process noise covariance matrix. Shape: (dims, dims) (positive semi-definite matrix)
        - `R`: measurement noise covariance matrix. Shape: (1, 1) (non-negative scalar)
        '''
        if Q is None:
            Q = np.zeros((self.dims, self.dims))
        if R is None:
            R = np.array([[0.0]])

        if Q.shape != (self.dims, self.dims) or R.shape != (1, 1):
            raise ValueError('Noise covariance matrices Q and R have incorrect dimensions.')
        elif not np.allclose(Q, Q.T):
            raise ValueError('Process noise covariance matrix Q must be symmetric.')
        elif not np.all(np.linalg.eigvals(Q) >= 0) or R[0, 0] < 0:
            raise ValueError('Noise covariance matrices Q and R must be positive semidefinite.')
        else:
            self.Q = Q
            self.R = R

    def sample_process_noise(self, n: int = 1) -> np.ndarray:
        '''
        Returns sample(s) of process noise, drawn from its Gaussian distributions.
        
        ### Arguments
        #### Optional
        - `n` (int, default = 1): number of noise samples to generate.
        
        ### Returns
        - `np.ndarray`: process noise samples. Shape (`dims`, `n`) (array of column vectors)
        '''        

        w_proc = np.random.multivariate_normal(
            mean=np.zeros(self.dims), cov=self.Q, size=n, check_valid='raise').T  # shape (dims, n)
        
        return w_proc

    def integrate_dynamics(self, t_start: float, t_stop: float, dt: float, 
            method='numerical', hold_noise_const: bool = False) -> tuple[np.ndarray, np.ndarray]:
        '''
        Integrates the equations describing the plant system from `t_start` to `t_stop` inclusive, using
        a step size of `dt`. The initial state is `self.x` at time `t_start`.
        The control input is `self.u` at all time steps within the integration period.
        At the end, the state at `t_stop` is stored in `self.x`.
        The measured output is not computed here.

        There are two methods for integration: 'numerical' and 'analytic':

        - 'numerical': use the Runge-Kutta 4th order method (RK4) with fixed step size `dt`. 
        - 'analytic': use the exact solution of the linear system, calculated with a cached matrix exponential.
       
        ### Arguments
        #### Required
        - `t_start` (float): start time of integration
        - `t_stop` (float): stop time of integration
        - `dt` (float): step size for numerical integration
        #### Optional
        - `hold_noise_const` (bool, default = False): if True, sample once and use it across all time steps. 
        If False, sample new noise at each `dt`.
        
        ### Returns
        - `tuple[np.ndarray, np.ndarray]`: arrays of time points and state trajectory (including both endpoints).
        '''
    
        if method == 'numerical':

            # create array [t_start, t_start + dt, t_start + 2 dt, ..., t_stop]
            t_span = get_t_span(t_start, t_stop, dt)  # shape (num_steps,)
            num_steps = t_span.shape[0]

            # set empty arrays
            x_span = np.zeros((self.dims, num_steps))  # shape (dims, num_steps)

            # set initial state
            x_span[:, 0] = self.x.reshape(self.dims,)
        
            # sample noise in advance
            if hold_noise_const:
                w_proc = self.sample_process_noise(n=1)  # shape (dims, 1)
                w_proc = np.tile(w_proc, (1, num_steps))  # shape (dims, num_steps)
            else:
                w_proc = self.sample_process_noise(n=num_steps)

            # define the function on the RHS of the update equation: x' = Ax + Bu + w_proc
            x_dot = lambda x, u, w_proc: self.A @ x + self.B @ u + w_proc

            for i in range(num_steps - 1):

                # get t_i
                t_i = t_span[i]

                # get noise values at t_i
                w_proc_i = w_proc[:, i].reshape(self.dims, 1)

                # TODO: consider using the exact solution, which would be
                # x_{i+1} = exp(A dt) @ x_i + A^{-1} @ (exp(A dt) - I) @ (B u) + integrated noise term (Brownian motion?)
                # cache exp(A dt) and A^{-1} for efficiency

                # use Runge-Kutta 4th order method (RK4) with fixed step size dt_int
                # NOTE: dt may be different for the last step, so we use t_span[i + 1] - t_i instead of dt here
                # NOTE: the control input is constant across the frame, so we always use self.u
                dt_i = t_span[i + 1] - t_i
                x_i = x_span[:, i].reshape(self.dims, 1)  # shape (dims, 1)
                k1 = x_dot(x_i, self.u, w_proc_i)
                k2 = x_dot(x_i + 0.5 * k1 * dt_i, self.u, w_proc_i)
                k3 = x_dot(x_i + 0.5 * k2 * dt_i, self.u, w_proc_i)
                k4 = x_dot(x_i + k3 * dt_i, self.u, w_proc_i)
                x_dot_i = (k1 + 2 * k2 + 2 * k3 + k4) / 6  # x' at t_i, adjusted by RK4
                x_span[:, i + 1] = x_span[:, i] + x_dot_i.flatten() * dt_i  # shape (dims,)

            # store final state in plant attributes
            # this becomes the initial state for the next time span when `integrate_dyamics` is called again
            self.x = x_span[:, -1].reshape(self.dims, 1)

            return t_span, x_span
        
        elif method == 'analytic':

            if not hold_noise_const:
                raise NotImplementedError('Analytic integration with time-varying noise is not implemented yet.')
            
            # TODO: consider using the exact solution, which would be
            # x_{i+1} = exp(A dt) @ x_i + A^{-1} @ (exp(A dt) - I) @ (B u) + integrated noise term (Brownian motion?)
            # cache exp(A dt) and A^{-1} for efficiency

            # TODO: generalise to allow for noise - need to replace (B u) with (B u + w_proc) and then 
            # need to decide how to handle the integrated noise term (Wiener process/Brownian motion?)

            first_bit = (self.exp_A_t_span @ self.x)  # shape (num_steps, dims, 1)
            bu_plus_noise = self.B @ self.u + self.sample_process_noise(n=1)  # shape (dims, 1)
            exp_A_dt_minus_I = self.exp_A_t_span - np.eye(self.dims)  # shape (num_steps, dims, dims)
            second_bit = (exp_A_dt_minus_I @ self.A_inv @ bu_plus_noise)  # shape (num_steps, dims, 1)
            x_span = (first_bit + second_bit)[:, :, 0].T  # shape (dims, num_steps)

            # store final state in plant attributes
            # this becomes the initial state for the next time span when `integrate_dyamics` is called again
            self.x = x_span[:, -1].reshape(self.dims, 1)

            # shift t_span_0 forwards to start at t_start instead of 0
            t_span = self.t_span_0 + t_start

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
