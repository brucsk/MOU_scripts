#!/usr/bin/env python
# coding: utf-8

# New estimation for MOU process
# 

import numpy as np
import scipy.linalg as spl
import scipy.stats as stt
import matplotlib.pyplot as pp


###############################################################################
class MOUv2:
    """
    Description of the class and a summary of its parameters, attributes and
    methods.

    Parameters
    ----------
    n_nodes : integer
        Number of nodes in the network.
    J : ndarray of rank-2
        Jacobian matrix between the nodes. The diagonal corresponds to a vector
        of time constants. For off-diagonal elements, the first dimension 
        corresponds to target nodes and the second dimension to source nodes 
        (J_ij is from i to j).
    mu : ndarray of rank-1
        Mean vector of the inputs to the nodes.
    Sigma : ndarray of rank-2
        Covariance matrix of the inputs to the nodes (multivariate Wiener process).

    Methods
    -------
    get_C : Returns the connectivity matrix (off-diagonal elements of the
        Jacobian).

    get_tau_x : Returns the time constant (related to the diagonal of the
        Jacobian)

    fit : Fit the model to a time series (time x nodes). The previous parameters
        (connectivity, etc.) are erased.

    fit_LO : Fit method relying on Lyapunov optimization (gradient descent).

    fit_moments : Fit method with maximum likelihood.

    score : Returns the goodness of fit after the optimization.
    
    simulate : Simulate the activity of the MOU process determined by J, mu and
        Sigma.    
    """

    def __init__(self, C=None, tau_x=1.0, mu=0.0, Sigma=None,
                random_state=None):
        """Initialize self. See help(MOU) for further information.
        The reason for separating the diagonal and off-diagonal elements in
        the Jacobian comes from focusing on the connectivity matrix as a graph.
        """

        # SECURITY CHECKS AND ARRANGEMENTS FOR THE PARAMETERS
        # Construct Jacobian
        if C is None:
            # 10 nodes by default
            self.n_nodes = 10
            # unconnected network
            C_tmp = np.zeros([self.n_nodes, self.n_nodes], dtype=float)
        elif type(C) == np.ndarray:
            if (not C.ndim == 2) or (not C.shape[0] == C.shape[1]):
                raise TypeError("""Argument C in MOU constructor must be square 
                                matrix (2D).""")
            self.n_nodes = C.shape[0]
            C_tmp = C
        else:
            raise TypeError("""Only matrix accepted for argument C in MOU 
                            constructor.""")

        if np.isscalar(tau_x):
            if tau_x <= 0:
                raise ValueError("""Scalar argument tau_x in MOU constructor must 
                                be negative.""")
            else:
                tau_x_tmp = np.ones(self.n_nodes) * tau_x
        elif type(tau_x) == np.ndarray:
            if (not tau_x.ndim == 1) or (not tau_x.shape[0] == self.n_nodes):
                raise TypeError("""Vector argument tau_x in MOU constructor must
                                be of same size as diagonal of C.""")
            tau_x_tmp = np.copy(tau_x)
        else:
            raise TypeError("""Only scalar value or vector accepted for argument
                            tau_x in MOU constructor.""")

        self.J = -np.eye(self.n_nodes) / tau_x_tmp + C_tmp
        if np.any(np.linalg.eigvals(self.J)>0):
            print("""The constructed MOU process has a Jacobian with negative 
                  eigenvalues, corresponding to unstable dynamics.""")

        # Inputs
        if np.isscalar(mu):
            self.mu = mu
        elif type(mu) == np.ndarray:
            if (not mu.ndim == 1) or (not mu.shape[0] == self.n_nodes):
                raise TypeError("""Vector argument mu in MOU constructor must be 
                                of same size as diagonal of C.""")
            self.mu = mu
        else:
            raise TypeError("""Only scalar value or vector accepted for argument 
                            tau_x in MOU constructor.""")

        if Sigma is None:
            # uniform unit variance by default
            self.Sigma = np.eye(self.n_nodes, dtype=float)
        elif np.isscalar(Sigma):
            if not (Sigma>0):
                raise TypeError("""Scalar argument Sigma in MOU constructor must 
                                be non-negative (akin to variance).""")
            self.Sigma = np.eye(self.n_nodes, dtype=float)
        elif type(Sigma) == np.ndarray:
            if (not Sigma.ndim == 2) or (not Sigma.shape[0] == Sigma.shape[1]) \
                                   or (not Sigma.shape[0] == self.n_nodes):
                raise TypeError("""Matrix argument Sigma in MOU constructor must
                                be square and of same size as C.""")
            if (not np.all(Sigma == Sigma.T)) or np.any(np.linalg.eigvals(Sigma) < 0):
                raise ValueError("""Matrix argument Sigma in MOU constructor must 
                                 be positive semidefinite (hence symmetric).""")
            self.Sigma = Sigma
        else:
            raise TypeError("""Only scalar value or matrix accepted for argument
                            Sigma in MOU constructor.""")

        # Set seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)


    # METHODS FOR CLASS MOU ###################################################
    
    def fit(self, X, y=None, method='lyapunov', **kwargs):
        """
        Generic fit method to call the adequate specific method.

        Parameters
        ----------
        X : ndarray
            The timeseries data of the system to estimate, of shape:
            T (time points) x n_nodes (numer of variables, e.g. number of ROIs).
        y : (for compatibility, not used here).
        method : string (optional)
            Set the optimization method; should be 'lyapunov' or 'moments'.

        Returns
        -------
        J : ndarray of rank 2
            The estimated Jacobian. Shape [n_nodes, n_nodes]
        Sigma : ndarray of rank 2
            Estimated input noise covariance. Shape [n_nodes, n_nodes]
        d_fit : dictionary
            A dictionary with diagnostics of the fit. Keys are: ['iterations',
            'distance', 'correlation'].
        """
        
        if (not type(X) == np.ndarray) or (not X.ndim == 2):
            raise TypeError("""Argument X must be matrix (time x nodes).""")
        n_T, self.n_nodes = np.shape(X)

        # Make sure a correct optimization method was entered
        if method not in ['lyapunov', 'moments']:
            raise ValueError("""Please enter a valid method: 'lyapunov' or 'moments'.""")

        # Decide number of time shifts for covariances Q_emp
        if 'i_tau_opt' in kwargs.keys():
            if (not type(kwargs['i_tau_opt']) == int):
                raise TypeError("""Argument Xi_tau_opt must be integer.""")
            # calculate time lags from 0 up to i_tau_opt
            n_tau = int(kwargs['i_tau_opt']) + 1
        else:
            n_tau = 2
            
        # Create a dictionary to store the diagnostics of fit
        self.d_fit = dict()
        self.d_fit['n_tau'] = n_tau

        # Lag (time-shifted) FC matrices
        Q_emp = np.zeros([n_tau, self.n_nodes, self.n_nodes], dtype=float)
        # Remove mean in the time series
        centered_X = X - X.mean(axis=0)
        # Calculate the lagged FC matrices
        n_T_span = n_T - n_tau + 1
        for i_tau in range(n_tau):
            Q_emp[i_tau] = np.tensordot(centered_X[0:n_T_span], \
                                        centered_X[i_tau:n_T_span+i_tau], \
                                        axes=(0,0))
        Q_emp /= float(n_T_span - 1)

        # Call adequate method for optimization and check for specific arguments
        if method == 'lyapunov':
            return self.fit_LO(Q_emp, **kwargs)
        elif method == 'moments':
            return self.fit_moments(Q_emp[0,:,:], Q_emp[1,:,:])

    def fit_Q(self, Q_emp, y=None, method='lyapunov', **kwargs):
        """
        New function to replace fit, to use Q0 and Q1 directly as the input
        instead of calculating them from the time series.

        Parameters
        ----------
        Q_emp : ndarray
            The empirical covariance matrices to fit, of shape: n_tau x n_nodes x n_nodes.
        y : (for compatibility, not used here).
        method : string (optional)
            Set the optimization method; should be 'lyapunov' or 'moments'.

        Returns
        -------
        J : ndarray of rank 2
            The estimated Jacobian. Shape [n_nodes, n_nodes]
        Sigma : ndarray of rank 2
            Estimated input noise covariance. Shape [n_nodes, n_nodes]
        d_fit : dictionary
            A dictionary with diagnostics of the fit. Keys are: ['iterations',
            'distance', 'correlation'].
        """
        
        if (not type(Q_emp) == np.ndarray) or (not Q_emp.ndim == 3):
            raise TypeError("""Argument Q_emp must be a 3D array (n_tau x n_nodes x n_nodes).""")
        n_tau, self.n_nodes = Q_emp.shape[0], Q_emp.shape[1]

        # Make sure a correct optimization method was entered
        if method not in ['lyapunov', 'moments']:
            raise ValueError("""Please enter a valid method: 'lyapunov' or 'moments'.""")

        # Decide number of time shifts for covariances Q_emp
        if 'i_tau_opt' in kwargs.keys():
            if (not type(kwargs['i_tau_opt']) == int):
                raise TypeError("""Argument Xi_tau_opt must be integer.""")
            # calculate time lags from 0 up to i_tau_opt
            n_tau = int(kwargs['i_tau_opt']) + 1
        else:
            n_tau = 2
            
        # Create a dictionary to store the diagnostics of fit
        self.d_fit = dict()
        self.d_fit['n_tau'] = n_tau

        # Call adequate method for optimization and check for specific arguments
        if method == 'lyapunov':
            return self.fit_LO(Q_emp, **kwargs)
        elif method == 'moments':
            return self.fit_moments(Q_emp[0,:,:], Q_emp[1,:,:])

        
    def fit_LO(self, Q_obj, i_tau_opt=1, mask_C=None, mask_Sigma=None, 
            epsilon_C=0.0001, epsilon_Sigma=0.01, regul_C=0.0, regul_Sigma=0.0, 
            min_val_C=0.0, max_val_C=1.0, min_val_Sigma_diag=0.0, max_iter=10000, 
            min_iter=10, algo_version='NeNe2020', track_eigs=False, eigs_every=1,
            eigs_store_full=False, track_cons_Q0_Q1 = True, verbose=False, **kwargs):
        """
        Estimation of MOU parameters (connectivity C, noise covariance Sigma,
        and time constant tau_x) with Lyapunov optimization as in: Gilson et al.
        Plos Computational Biology (2016).

        Parameters
        ----------
        Q_obj : ndarray
            The covariance matrix
        i_tau_opt : integer (optional)
            Mask of known non-zero values for connectivity matrix, for example
            estimated by DTI.
        mask_C : boolean ndarray of rank-2 (optional)
            Mask of known non-zero values for connectivity matrix, for example
            estimated by DTI. By default, all connections are allowed (except
            self-connections)
        mask_Sigma : boolean ndarray of rank-2 (optional)
            Mask of known non-zero values for the input covariance matrix
            (by default diagonal).
        epsilon_C : float (optional)
            Learning rate for connectivity (this should be about n_nodes times
            smaller than epsilon_Sigma).
        epsilon_Sigma : float (optional)
            Learning rate for Sigma (this should be about n_nodes times larger
            than epsilon_EC).
        regul_C : float (optional)
            Regularization parameter for connectivity. Try first a value of 0.5.
        regul_Sigma : float (optional)
            Regularization parameter for Sigma. Try first a value of 0.001.
        min_val_C : float (optional)
            Minimum value to bound connectivity estimate. This should be zero
            or slightly negative (too negative limit can bring to an inhibition
            dominated system). If the empirical covariance has many negative
            entries then a slightly negative limit can improve the estimation
            accuracy.
        max_val_C : float (optional)
            Maximum value to bound connectivity estimate. This is useful to
            avoid large weight that make the system unstable. If the estimated
            connectivity saturates toward this value (it usually doesn't happen)
            it can be increased.
        max_iter : integer (optional)
            Number of maximum optimization steps. If final number of iterations
            reaches this maximum it means the algorithm has not converged.
        min_iter : integer (optional)
            Number of minimum optimization steps before testing if end of 
            optimization (increase of model error).
        algo_version : string (optional)
            Version of algorithm for the optimization of the network weights.
        track_eigs : boolean (optional)
            Whether to track the eigenvalues of the Jacobian during optimization.
        eigs_every : integer (optional)
            If track_eigs is True, the frequency (in number of iterations) to track the eigenvalues.
        eigs_store_full : boolean (optional)
            If track_eigs is True, whether to store the full eigenvalues at each tracking step (can consume a lot of memory).
        track_cons_Q0_Q1 : boolean (optional)
            Whether to track the consistency of Q0 and Q1 during optimization, that is, for J, we have Q1 ~ Q0 * expm(J).

        Returns
        -------
        J : ndarray of rank 2
            The estimated Jacobian. Shape [n_nodes, n_nodes]
        Sigma : ndarray of rank 2
            Estimated noise covariance. Shape [n_nodes, n_nodes]
        d_fit : dictionary
            A dictionary with diagnostics of the fit. Keys are: ['iterations',
            'distance', 'correlation'].
        """
        # TODO: make better graphics (deal with axes separation, etc.)

        if (not type(i_tau_opt) == int) or (i_tau_opt <= 0):
            raise ValueError('Scalar value i_tau_opt must be non-zero')
            
        # select the update
        if algo_version == 'PCB2016':
            # update published in PLoS Comput Biol 2016
            def fun_Delta_J(J_opt, Q0, Qtau, Delta_Q0, Delta_Qtau):
                return np.dot( np.linalg.pinv(Q0), Delta_Q0 + np.dot( Delta_Qtau, spl.expm(-J_opt) ) )
        elif algo_version == 'NeNe2020':
            # update published in Netw Neurosci 2020
            def fun_Delta_J(J_opt, Q0, Qtau, Delta_Q0, Delta_Qtau):
                return np.dot( np.linalg.pinv(Q0), Delta_Q0 ) + np.dot( Delta_Q0, np.linalg.pinv(Q0) ) \
                    + np.dot( np.linalg.pinv(Qtau), Delta_Qtau ) + np.dot( Delta_Qtau, np.linalg.pinv(Qtau) )
        elif algo_version == 'true':
            # update
            def fun_Delta_J(J_opt, Q0, Qtau, Delta_Q0, Delta_Qtau):
                return np.dot( np.linalg.pinv(Q0), - c0 * Delta_Q0 + c1 * np.dot( Delta_Qtau, spl.expm(-J_opt) ) ) 
# TRUE                return np.dot( np.linalg.pinv(Q0), - c0 * Delta_Q0 + c1 * np.dot( Delta_Qtau, spl.expm(-J_opt) ) ) \
#                     + np.dot( np.linalg.pinv(Qtau), - c0 * np.dot(Delta_Q0, spl.expm(J_opt)) + c1 * Delta_Qtau ) 
#                return np.dot( np.linalg.pinv(Q0), - 0.2 * Delta_Q0 + np.dot( Delta_Qtau, spl.expm(-J_opt) ) ) \
#                     + np.dot( np.linalg.pinv(Qtau), - 0.2 * np.dot(Delta_Q0, spl.expm(J_opt)) + Delta_Qtau ) 
#                return np.dot( np.linalg.pinv(Q0), - 0.1 * Delta_Q0 + np.dot( Delta_Qtau, spl.expm(-J_opt) ) ) \
#                return np.dot( np.linalg.pinv(Q0), - np.dot( Delta_Q0, spl.expm(J_opt) ) + Delta_Qtau ) WRONG TRUE
        else:
            raise ValueError('unkown version for optimization algorithm of fit_LO')


        # Objective FC matrices (empirical)
        Q0_obj = Q_obj[0]
        Qtau_obj = Q_obj[i_tau_opt]

        # Autocovariance time constant (exponential decay)
        log_ac = np.log( np.maximum( Q_obj.diagonal(axis1=1,axis2=2), 1e-10 ) )
        v_tau = np.arange(Q_obj.shape[0], dtype=float)
        lin_reg = np.polyfit( np.repeat(v_tau, self.n_nodes), log_ac.reshape(-1), 1 )
        tau_obj = -1.0 / lin_reg[0]

        # coefficients to balance the model error between Q0 and Qtau
        norm_Q0_obj = np.linalg.norm(Q0_obj)
        norm_Qtau_obj = np.linalg.norm(Qtau_obj)
        c0 = norm_Qtau_obj/(norm_Q0_obj+norm_Qtau_obj)
        c1 = 1.0 - c0

        # mask for existing connections for EC and Sigma
        mask_diag = np.eye(self.n_nodes, dtype=bool)
        if mask_C is None:
            # Allow all possible connections to be tuned except self-connections (on diagonal)
            mask_C = np.logical_not(mask_diag)
        if mask_Sigma is None:
            # Independent noise (no cross-covariances for Sigma)
            mask_Sigma = np.eye(self.n_nodes, dtype=bool)

        # Initialise network and noise. Give initial parameters
        C = np.zeros([self.n_nodes, self.n_nodes], dtype=float)
        tau_x = np.copy(tau_obj)
#        Sigma = np.eye(self.n_nodes, dtype=float)
        Sigma = 2 / tau_obj * Q0_obj
        Sigma[np.logical_not(mask_diag)] = 0

        # Best distance between model and empirical data
        best_dist = 1e10
        best_Pearson = 0.0

        # Arrays to record model parameters and outputs
        # model error = matrix distance between FC matrices
        dist_Q_hist = np.zeros([max_iter], dtype=float)
        # Pearson correlation between model and objective FC matrices
        Pearson_Q_hist = np.zeros([max_iter], dtype=float)

        # Store eigenvalues if tracking is enabled
        if track_eigs:
            eig_iter = []
            eig_max_real = []
            eig_min_real = []

            if eigs_store_full:
                eigs_history = []

        if track_cons_Q0_Q1:
            cons_Q0_Q1_hist = []

        # identity matrix
        id_mat = np.eye(self.n_nodes, dtype=float)

        # run the optimization process
        stop_opt = False
        i_iter = 0
        while not stop_opt:

            # calculate Jacobian of dynamical system
            J_opt = -id_mat / tau_x + C

            # compute eigenvalues of Jacobian 
            eigvals_J_opt = np.linalg.eigvals(J_opt)
            # store eigenvalues if tracking is enabled
            if track_eigs and np.mod(i_iter, eigs_every) == 0:
                eig_iter.append(i_iter)
                eig_max_real.append(np.max(eigvals_J_opt.real))
                eig_min_real.append(np.min(eigvals_J_opt.real))
                if eigs_store_full:
                    eigs_history.append(eigvals_J_opt)

            # Track consistency between data-given Q0 and Q1 for the current J (matrix distance) 
            if track_cons_Q0_Q1:
                # Calculate consistency between Q0 and Q1 for the current J (matrix distance)
                cons_Q0_Q1_hist.append(np.linalg.norm(Qtau_obj - np.dot(Q0_obj, spl.expm(J_opt * i_tau_opt))) / np.linalg.norm(Qtau_obj))
                if verbose:
                    print("Consistency between Q0 and Q1 for current J (Q1 - Q0*exp(J)/Q1):", cons_Q0_Q1_hist[-1])

            # Calculate Q0 and Qtau for model
            Q0 = spl.solve_continuous_lyapunov(J_opt.T, -Sigma)
            Qtau = np.dot( Q0, spl.expm( J_opt * i_tau_opt ) )

            # difference matrices between model and objectives
            Delta_Q0 = Q0_obj - Q0
            Delta_Qtau = Qtau_obj - Qtau

            # Calculate error between model and empirical data for Q0 and FC_tau (matrix distance)
            dist_Q0 = np.linalg.norm(Delta_Q0) / norm_Q0_obj
            dist_Qtau = np.linalg.norm(Delta_Qtau) / norm_Qtau_obj
            dist_Q_hist[i_iter] = 0.5 * (dist_Q0 + dist_Qtau)

            # Calculate corr between model and empirical data for Q0 and FC_tau
            Pearson_Q0 = stt.pearsonr( Q0.reshape(-1), Q0_obj.reshape(-1) )[0]
            Pearson_Qtau = stt.pearsonr( Qtau.reshape(-1), Qtau_obj.reshape(-1) )[0]
            Pearson_Q_hist[i_iter] = 0.5 * (Pearson_Q0  + Pearson_Qtau)

            # Best fit given by best Pearson correlation coefficient
            # for both Q0 and Qtau (better than matrix distance)
            if dist_Q_hist[i_iter] < best_dist:
                best_dist = dist_Q_hist[i_iter]
                best_Pearson = Pearson_Q_hist[i_iter]
                J_best = np.copy(J_opt)
                Sigma_best = np.copy(Sigma)
            else:
                # wait at least 5 optimization steps before stopping
                stop_opt = i_iter > min_iter

            # Jacobian update with weighted FC updates depending on respective error
            Delta_J = fun_Delta_J(J_opt, Q0, Qtau, Delta_Q0, Delta_Qtau)
#            Delta_J = np.dot( np.linalg.pinv(Q0), Delta_Q0 ) + np.dot( Delta_Q0, np.linalg.pinv(Q0) ) \
#                    + np.dot( np.linalg.pinv(Qtau), Delta_Qtau ) + np.dot( Delta_Qtau, np.linalg.pinv(Qtau) )
    
            # Update effective conectivity matrix (regularization is L2)
            C[mask_C] += epsilon_C * ( Delta_J - regul_C * C )[mask_C]
            C[mask_C] = np.clip(C[mask_C], min_val_C, max_val_C)

            # Update noise matrix Sigma (regularization is L2)
            Delta_Sigma = - np.dot(J_opt.T, Delta_Q0) - np.dot(Delta_Q0, J_opt)
            Sigma[mask_Sigma] += epsilon_Sigma * ( Delta_Sigma - regul_Sigma * Sigma )[mask_Sigma]
            Sigma[mask_diag] = np.maximum(Sigma[mask_diag], min_val_Sigma_diag)

            # display optimization evolution if verbose==True
            if verbose: # and np.mod(i_iter, 10)==0:
                print('optimisation step:', i_iter, '; model error =', dist_Q_hist[i_iter])
                #print('max real part of eigenvalues:', np.max(eigvals_J_opt.real), '; min real part of eigenvalues:', np.min(eigvals_J_opt.real))
            # Check if max allowed number of iterations have been reached
            if i_iter >= max_iter-1:
                stop_opt = True
                print('Optimization did not converge. Maximum number of iterations arrived.')
            # Check if iteration has finished or still continues
            if stop_opt:
                self.d_fit['iterations'] = i_iter+1
                self.d_fit['distance'] = best_dist
                self.d_fit['correlation'] = best_Pearson
                self.d_fit['distance history'] = dist_Q_hist
                self.d_fit['correlation history'] = Pearson_Q_hist
                # test
                self.d_fit['distFC0'] = dist_Q0
                self.d_fit['distFC1'] = dist_Qtau
                # add eigenvalues diagnostics if tracking is enabled
                if track_eigs:
                    self.d_fit['eig_iter'] = eig_iter
                    self.d_fit['eig_max_real'] = eig_max_real
                    self.d_fit['eig_min_real'] = eig_min_real
                    if eigs_store_full:
                        self.d_fit['eigs_history'] = eigs_history
                if track_cons_Q0_Q1:
                    self.d_fit['cons_Q0_Q1_history'] = cons_Q0_Q1_hist
            else:
                i_iter += 1

        # Save the results and return
        self.J = J_best # matrix
        self.Sigma = Sigma_best # matrix

        return self

    
    def fit_moments(self, Q0_obj, Q1_obj, mask_C=None):
        """
        Estimation of MOU parameters (connectivity C, noise covariance Sigma,
        and time constant tau_x) with moments method.

        Parameters
        ----------
        Q0_obj : ndarray of rank 2
            The zero-lag covariance matrix of the time series to fit.
        Q1_obj : ndarray of rank 2
            The 1-lag covariance matrix of the time series to fit.
        mask_C : boolean ndarray of rank-2 (optional)
            Mask of known non-zero values for connectivity matrix, for example
            estimated by DTI.

        Returns
        -------
        J : ndarray of rank 2
            The estimated Jacobian. Shape [n_nodes, n_nodes]
        Sigma : ndarray of rank 2
            Estimated noise covariance. Shape [n_nodes, n_nodes]
        d_fit : dictionary
            A dictionary with diagnostics of the fit. Keys are: ['iterations',
            'distance', 'correlation'].
        """
        # Jacobian estimate
        inv_Q0 = np.linalg.inv(Q0_obj)
        J = spl.logm( np.dot(inv_Q0, Q1_obj) )
        # Sigma estimate
        Sigma = - np.dot(J.conjugate(), Q0_obj) - np.dot(Q0_obj, J)

        # masks for existing positions
        mask_diag = np.eye(self.n_nodes, dtype=np.bool)
        if mask_C is None:
            # Allow all possible connections to be tuned except self-connections (on diagonal)
            mask_C = np.logical_not(mask_diag)
            
        # cast to real matrices
        if np.any(np.iscomplex(J)):
            print("Warning: complex values in J; casting to real!")
        J_best = np.real(J)
        J_best[np.logical_not(np.logical_or(mask_C,mask_diag))] = 0
        if np.any(np.iscomplex(Sigma)):
            print("Warning: complex values in Sigma; casting to real!")
        Sigma_best = np.real(Sigma)

        # model theoretical covariances with real J and Sigma
        Q0 = spl.solve_continuous_lyapunov(J_best.T, -Sigma_best)
        Q1 = np.dot( Q0, spl.expm(J_best) )

        # Calculate error between model and empirical data for Q0 and FC_tau (matrix distance)
        dist_Q0 = np.linalg.norm(Q0 - Q0_obj) / np.linalg.norm(Q0_obj)
        dist_Qtau = np.linalg.norm(Q1 - Q1_obj) / np.linalg.norm(Q1_obj)
        self.d_fit['distance'] = 0.5 * (dist_Q0 + dist_Qtau)

        # Average correlation between empirical and theoretical
        Pearson_Q0 = stt.pearsonr( Q0.reshape(-1), Q0_obj.reshape(-1) )[0]
        Pearson_Qtau = stt.pearsonr( Q1.reshape(-1), Q1_obj.reshape(-1) )[0]
        self.d_fit['correlation'] = 0.5 * (Pearson_Q0 + Pearson_Qtau)

        # Save the results and return
        self.J = J_best # matrix
        self.Sigma = Sigma_best # matrix

        return self

    
    def score(self):
        """
        Returns the correlation between goodness of fit of the MOU to the 
        data, measured by the Pearson correlation between the obseved 
        covariances and the model covariances. 
        """
        try:
            return self.d_fit['correlation']
        except:
            print('The model should be fitted first.')
            return np.nan
            ## GORKA: Shall this raise a RunTimeWarning or other type of warning?

            
    def model_covariance(self, tau=0.0):
        """
        Calculates theoretical (lagged) covariances of the model given the
        parameters (forward step). Notice that this is not the empirical
        covariance matrix as estimated from simulated time series.

        Parameters
        ----------
        tau : scalar
            The time lag to calculate the covariance. It can be a positive or
            negative.

        Returns
        -------
        FC : ndarray of rank-2
            The (lagged) covariance matrix.
        """

        # Calculate zero lag-covariance Q0 by solving Lyapunov equation
        Q0 = spl.solve_continuous_lyapunov(self.J.T, -self.Sigma)
        # Calculate the effect of the lag (still valid for tau = 0.0)
        if tau >= 0.0:
            return np.dot(Q0, spl.expm(tau * self.J))
        else:
            return np.dot(spl.expm(-tau * self.J.T), Q0)

        
    def simulate(self, T=100, dt=0.05, sampling_step=1., random_state=None):
        """
        Simulate the MOU process with simple Euler integration defined by the
        time step.

        Parameters
        ----------
        T : integer (optional)
            Duration of simulation.
        dt : scalar (optional)
            Integration time step.
        sampling_step : scalar (optional)
            Period for subsampling the generated time series.
        random_state : long or int (optional)
            Description here ...

        Returns
        --------
        ts : ndarray of rank-2
            Time series of simulated network activity of shape [T, n_nodes]

        Notes
        -----
        It is possible to include an acitvation function to
        give non linear effect of network input; here assumed to be identity
        """
        # 0) SECURITY CHECKS
        if dt<0.:
            raise ValueError("Integration step has to be positive. dt<0 given.")
        if T<=dt:
            raise ValueError("Duration of simulation too short. T<dt given.")
        if sampling_step<dt:
            raise ValueError("Decrease dt or increase sampling_step. sampling_step<dt given.")
        # set seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)

        # 1) PREPARE FOR THE SIMULATION
        # 1.1) Simulation time
        # initial period to remove effect of initial conditions
        T0 = int(10. / (-self.J.diagonal()).max())
        # Sampling to get 1 point every second
        n_sampl = int(sampling_step / dt)
        # simulation time in discrete steps
        n_T = int( np.ceil(T / dt) )
        n_T0 = int(T0 / dt)

        # 1.2) Initialise the arrays
        # Array for generated time-series
        ts = np.zeros([int(n_T/n_sampl), self.n_nodes], dtype=float)
        # Set initial conditions for activity
        x_tmp = np.random.normal(size=[self.n_nodes])
        # Generate noise for all time steps before simulation
        noise = np.random.normal(size=[n_T0+n_T, self.n_nodes], scale=(dt**0.5))

        # 2) RUN THE SIMULATION
        # rescaled inputs and Jacobian with time step of simulation
        mu_dt = self.mu * dt
        J_dt = self.J * dt
        # calculate square root matrix of Sigma
        sqrt_Sigma = spl.sqrtm(self.Sigma)
        for t in np.arange(n_T0+n_T):
            # update of activity
            x_tmp += np.dot(x_tmp, J_dt) + mu_dt + np.dot(sqrt_Sigma, noise[t])
            # Discard first n_T0 timepoints
            if t >= n_T0:
                # Subsample timeseries (e.g. to match fMRI time resolution)
                if np.mod(t-n_T0,n_sampl) == 0:
                    # save the result into the array
                    ts[int((t-n_T0)/n_sampl)] = x_tmp

        return ts

    def simulateBD(self, B, D, T=100, dt=0.05, sampling_step=1., random_state=None):
        """
        Simulate the MOU process with simple Euler integration defined by the
        time step.

        Parameters
        ----------
        T : integer (optional)
            Duration of simulation.
        dt : scalar (optional)
            Integration time step.
        sampling_step : scalar (optional)
            Period for subsampling the generated time series.
        random_state : long or int (optional)
            Description here ...

        Returns
        --------
        ts : ndarray of rank-2
            Time series of simulated network activity of shape [T, n_nodes]

        Notes
        -----
        It is possible to include an acitvation function to
        give non linear effect of network input; here assumed to be identity
        """
        # 0) SECURITY CHECKS
        if dt<0.:
            raise ValueError("Integration step has to be positive. dt<0 given.")
        if T<=dt:
            raise ValueError("Duration of simulation too short. T<dt given.")
        if sampling_step<dt:
            raise ValueError("Decrease dt or increase sampling_step. sampling_step<dt given.")
        # set seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)

        # 1) PREPARE FOR THE SIMULATION
        # 1.1) Simulation time
        # initial period to remove effect of initial conditions
        J=-B
        Sigma=2*D
        T0 = int(10. / (-J.diagonal()).max())
        # Sampling to get 1 point every second
        n_sampl = int(sampling_step / dt)
        # simulation time in discrete steps
        n_T = int( np.ceil(T / dt) )
        n_T0 = int(T0 / dt)

        # 1.2) Initialise the arrays
        # Array for generated time-series
        ts = np.zeros([int(n_T/n_sampl), self.n_nodes], dtype=float)
        # Set initial conditions for activity
        x_tmp = np.random.normal(size=[self.n_nodes])
        # Generate noise for all time steps before simulation
        noise = np.random.normal(size=[n_T0+n_T, self.n_nodes], scale=(dt**0.5))

        # 2) RUN THE SIMULATION
        # rescaled inputs and Jacobian with time step of simulation
        mu_dt = self.mu * dt
        J_dt = J * dt
        # calculate square root matrix of Sigma
        sqrt_Sigma = spl.sqrtm(self.Sigma)
        for t in np.arange(n_T0+n_T):
            # update of activity
            x_tmp += np.dot(x_tmp, J_dt) + mu_dt + np.dot(sqrt_Sigma, noise[t])
            # Discard first n_T0 timepoints
            if t >= n_T0:
                # Subsample timeseries (e.g. to match fMRI time resolution)
                if np.mod(t-n_T0,n_sampl) == 0:
                    # save the result into the array
                    ts[int((t-n_T0)/n_sampl)] = x_tmp

        return ts



# ## Simulation

# In[2]:
#
#
# N = 20 # number of nodes
# d = 0.3 # density of connectivity
## generate random matrix
# C_orig = tools.make_rnd_connectivity(N, density=d, w_min=0.5/N/d, w_max=1.2/N/d)
#
## create MOU process
# mou_orig = MOU(C_orig)
#
# use_topology = True
#
# T = 1000 # time in seconds
## simulate
# ts_sim = mou_orig.simulate(T)
#
## plots
# pp.figure()
# pp.plot(range(T),ts_sim)
# pp.xlabel('time')
# pp.ylabel('activity')
# pp.title('simulated MOU signals')
# #
# D = np.linalg.eigvals(C_orig)
# pp.figure()
# pp.scatter(np.real(D),np.imag(D))
# pp.plot([1,1],[-1,1],'--k')
# pp.xlabel('real part of eigenvalue')
# pp.ylabel('imag part of eigenvalue')
# pp.title('spectrum of original C')

# pp.show()

#
## In[ ]:
#
#
#Q_sim = np.tensordot(ts_sim,ts_sim,axes=(0,0)) / (T-1)
#
#J = -np.eye(N) + C_orig
#Sigma = np.eye(N)
#Q_th = spl.solve_continuous_lyapunov(J,-Sigma)
#
#
## plots
#pp.figure()
#pp.imshow(Q_sim)
#pp.colorbar()
#pp.xlabel('target ROI')
#pp.ylabel('source ROI')
#pp.title('covariance matrix (functional connectivity)')
#
#pp.figure()
#pp.plot([0,Q_th.max()],[0,Q_th.max()],'--k')
#pp.plot(Q_sim,Q_th,'.b')
#pp.xlabel('simulated covariances')
#pp.ylabel('theoretical covariances')
#
#pp.show()
#
#
## ## Connectivity estimation
#
## In[ ]:
#
#
## Lyapunov optimization
#mou_est_2020 = MOU()
#if not use_topology:
#    # estimate of weights without knowledge of the topology of existing weights in C_orig
#    # regularization may be helpful here to "push" small weights to zero here
#    mou_est_2020.fit(ts_sim, i_tau_opt=1, algo_version='NeNe2020')
##    mou_est_2020.fit(ts_sim, i_tau_opt=1, regul_C=1., algo_version='NeNe2020')
#else:
#    # estimate of weights knowing the topology of existing weights in C_orig
#    mou_est_2020.fit(ts_sim, i_tau_opt=1, mask_C=C_orig>0, algo_version='NeNe2020')
#
#C_est_2020 = mou_est_2020.get_C()
#
#
## plots
#pp.figure()
#pp.imshow(C_orig,vmin=0)
#pp.colorbar()
#pp.xlabel('target ROI')
#pp.ylabel('source ROI')
#pp.title('original connectivity')
#
#pp.figure()
#pp.imshow(C_est_2020,vmin=0)
#pp.colorbar()
#pp.xlabel('target ROI')
#pp.ylabel('source ROI')
#pp.title('estimated connectivity')
#
#pp.figure()
#pp.plot([0,C_orig.max()],[0,C_orig.max()],'--k')
#pp.plot(C_est_2020,C_orig,'xr')
#pp.xlabel('estimated connectivity')
#pp.ylabel('original connectivity')
#
#pp.show()
#
#
## In[ ]:
#
#
## Lyapunov optimization 2016
#mou_est_2016 = MOU()
#if not use_topology:
#    # estimate of weights without knowledge of the topology of existing weights in C_orig
#    # regularization may be helpful here to "push" small weights to zero here
#    mou_est_2016.fit(ts_sim, i_tau_opt=1, algo_version='true')
#else:
#    # estimate of weights knowing the topology of existing weights in C_orig
#    mou_est_2016.fit(ts_sim, i_tau_opt=1, mask_C=C_orig>0, algo_version='true')
#
#C_est_2016 = mou_est_2016.get_C()
#
#
## plots
#pp.figure()
#pp.imshow(C_orig,vmin=0)
#pp.colorbar()
#pp.xlabel('target ROI')
#pp.ylabel('source ROI')
#pp.title('original connectivity')
#
#pp.figure()
#pp.imshow(C_est_2016,vmin=0)
#pp.colorbar()
#pp.xlabel('target ROI')
#pp.ylabel('source ROI')
#pp.title('estimated connectivity')
#
#pp.figure()
#pp.plot([0,C_orig.max()],[0,C_orig.max()],'--k')
#pp.plot(C_est_2016,C_orig,'xr')
#pp.xlabel('estimated connectivity')
#pp.ylabel('original connectivity')
#
#pp.show()
#
#
## In[ ]:
#
#
## moments method
#mou_est_mom = MOU()
#mou_est_mom.fit(ts_sim, method='moments')
#
#C_est_mom = mou_est_mom.get_C()
#
#
## plots
#pp.figure()
#pp.imshow(C_orig,vmin=0)
#pp.colorbar()
#pp.xlabel('target ROI')
#pp.ylabel('source ROI')
#pp.title('original connectivity')
#
#pp.figure()
#pp.imshow(C_est_mom,vmin=0)
#pp.colorbar()
#pp.xlabel('target ROI')
#pp.ylabel('source ROI')
#pp.title('estimated connectivity')
#
#pp.figure()
#pp.plot([0,C_orig.max()],[0,C_orig.max()],'--k')
#pp.plot(C_est_mom,C_orig,'xr')
#pp.xlabel('estimated connectivity')
#pp.ylabel('original connectivity')
#
#pp.show()
#
#
## In[ ]:
#
#
#print('model fit for Lyap 2020:',mou_est_2020.d_fit['correlation'])
#print('model fit for lyap 2016:',mou_est_2016.d_fit['correlation'])
#print('model fit for moments:',mou_est_mom.d_fit['correlation'])
#
#print('C_orig fit for Lyap 2020',stt.pearsonr(C_orig.flatten(),C_est_2020.flatten()))
#print('C_orig fit for Lyap 2016',stt.pearsonr(C_orig.flatten(),C_est_2016.flatten()))
#print('C_orig fit for moments',stt.pearsonr(C_orig.flatten(),C_est_mom.flatten()))
#print('C_orig fit for moments + positive constraints',stt.pearsonr(C_orig.flatten(),np.maximum(C_est_mom,0).flatten()))
#
#min_weight = min(C_est_2020.min(),C_est_mom.min())
#max_weight = max(C_est_2020.max(),C_est_mom.max())
#bins = np.linspace(min_weight,max_weight,40)
#
#pp.figure()
#pp.subplot(311)
#pp.hist(C_est_2020[C_orig>0], bins=bins, histtype='step', color='g')
#pp.hist(C_est_2020[C_orig==0], bins=bins, histtype='step', color='k')
#pp.title('green = true; black = false')
#pp.subplot(312)
#pp.hist(C_est_2016[C_orig>0], bins=bins, histtype='step', color='g')
#pp.hist(C_est_2016[C_orig==0], bins=bins, histtype='step', color='k')
#pp.ylabel('distributions estimates')
#pp.subplot(313)
#pp.hist(C_est_mom[C_orig>0], bins=bins, histtype='step', color='g')
#pp.hist(C_est_mom[C_orig==0], bins=bins, histtype='step', color='k')
#
#pp.figure()
#pp.subplot(311)
#pp.hist(C_est_2020[C_orig>0], bins=bins, histtype='step', cumulative=True, density=True, color='g')
#pp.hist(C_est_2020[C_orig==0], bins=bins, histtype='step', cumulative=True, density=True, color='k')
#pp.title('green = true; black = false')
#pp.subplot(312)
#pp.hist(C_est_2016[C_orig>0], bins=bins, histtype='step', cumulative=True, density=True, color='g')
#pp.hist(C_est_2016[C_orig==0], bins=bins, histtype='step', cumulative=True, density=True, color='k')
#pp.ylabel('cumulative density')
#pp.subplot(313)
#pp.hist(C_est_mom[C_orig>0], bins=bins, histtype='step', cumulative=True, density=True, color='g')
#pp.hist(C_est_mom[C_orig==0], bins=bins, histtype='step', cumulative=True, density=True, color='k')
#
#pp.figure()
#pp.plot([0,C_orig.max()],[0,C_orig.max()],'--k')
#pp.plot(C_est_2020,C_est_mom,'xr')
#pp.xlabel('LO 2020 estimate')
#pp.ylabel('moment estimate')
#
#pp.figure()
#pp.plot([0,C_orig.max()],[0,C_orig.max()],'--k')
#pp.plot(C_est_2020,C_est_2016,'xr')
#pp.xlabel('LO 2020 estimate')
#pp.ylabel('LO 2016 estimate')
#
#pp.show()
#
#
##%% test asym
#
#def asym(M):
#    return np.abs(M-M.T).sum() / np.abs(M).sum() / 2
#
#mask_offdiag = np.logical_not(np.eye(N, dtype=np.bool))
#
#print('asym C orig:', asym(C_orig))
#print('asym C lyap 2020:', asym(C_est_2020))
#print('asym C lyap 2016:', asym(C_est_2016))
#print('asym C mom:', asym(C_est_mom))


#%% Load .mat data and fit MOU model

# Load the .mat file
import scipy.io as sio
mat_data = sio.loadmat('SC_EnigmadK68.mat')

# Extract the data tensor (adjust key name based on actual structure)
# Common keys in .mat files: try to find the data
data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
print(f"Available keys in .mat file: {data_keys}")

# Assuming the main data is in one of the non-private keys
# You may need to adjust this based on the actual key name
if len(data_keys) == 1:
    X_data = mat_data[data_keys[0]]
else:
    # Try common names or take the first non-metadata key
    possible_names = ['data', 'X', 'timeseries', 'ts', 'SC']
    X_data = None
    for name in possible_names:
        if name in mat_data:
            X_data = mat_data[name]
            break
    if X_data is None:
        X_data = mat_data[data_keys[0]]

print(f"Data shape: {X_data.shape}")

print(mat_data)
# Ensure data is in the correct shape (t x n_nodes)
if X_data.shape[0] < X_data.shape[1]:
    print("Transposing data to ensure shape is (t x n_nodes)")
    X_data = X_data.T

n_timepoints, n_nodes = X_data.shape
print(f"Time points: {n_timepoints}, Nodes: {n_nodes}")

# Initialize and fit MOU model
mou_fitted = MOUv2()
mou_fitted.fit(X_data, method='lyapunov', i_tau_opt=1)

print(f"\nModel fit correlation: {mou_fitted.score():.4f}")
print(f"Model fit distance: {mou_fitted.d_fit['distance']:.4f}")
print(f"Number of iterations: {mou_fitted.d_fit['iterations']}")

# Get the fitted parameters
J_fitted = mou_fitted.J
Sigma_fitted = mou_fitted.Sigma

print(f"\nFitted Jacobian eigenvalues (max real part): {np.real(np.linalg.eigvals(J_fitted)).max():.4f}")


#%% Compute entropy of the fitted MOU process

def compute_mou_entropy(Sigma, Q0=None, J=None):
    """
    Compute the differential entropy of a multivariate Ornstein-Uhlenbeck process.
    
    For a multivariate Gaussian with covariance Q0, the differential entropy is:
    H = 0.5 * n * (1 + log(2*pi)) + 0.5 * log(det(Q0))
    
    Parameters
    ----------
    Sigma : ndarray
        Noise covariance matrix
    Q0 : ndarray, optional
        Steady-state covariance matrix. If not provided, will be computed from J and Sigma.
    J : ndarray, optional
        Jacobian matrix (required if Q0 is not provided)
    
    Returns
    -------
    entropy : float
        Differential entropy in nats
    """
    
    if Q0 is None:
        if J is None:
            raise ValueError("Either Q0 or J must be provided")
        # Compute steady-state covariance from Lyapunov equation
        Q0 = spl.solve_continuous_lyapunov(J.T, -Sigma)
    
    n = Q0.shape[0]
    
    # Compute determinant
    sign, logdet = np.linalg.slogdet(Q0)
    
    if sign <= 0:
        print("Warning: Covariance matrix is not positive definite!")
        return np.nan
    
    # Differential entropy: H = 0.5 * n * (1 + log(2*pi)) + 0.5 * log(det(Q0))
    entropy = 0.5 * n * (1 + np.log(2 * np.pi)) + 0.5 * logdet
    
    return entropy


# Compute steady-state covariance
Q0_fitted = spl.solve_continuous_lyapunov(J_fitted.T, -Sigma_fitted)

# Compute entropy
entropy_fitted = compute_mou_entropy(Sigma_fitted, Q0=Q0_fitted)
print(f"\nDifferential entropy of fitted MOU process: {entropy_fitted:.4f} nats")
print(f"Differential entropy in bits: {entropy_fitted / np.log(2):.4f} bits")

# Additional entropy measures
# Entropy rate (for time-continuous process)
entropy_rate = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * Sigma_fitted))
print(f"Entropy rate: {entropy_rate:.4f} nats/time")
print(f"Entropy rate: {entropy_rate / np.log(2):.4f} bits/time")


#%% Visualization of fitted model

pp.figure(figsize=(15, 5))

pp.subplot(131)
pp.imshow(J_fitted, cmap='RdBu_r', aspect='auto')
pp.colorbar(label='Weight')
pp.title('Fitted Jacobian Matrix')
pp.xlabel('Source node')
pp.ylabel('Target node')

pp.subplot(132)
pp.imshow(Sigma_fitted, cmap='viridis', aspect='auto')
pp.colorbar(label='Noise covariance')
pp.title('Fitted Noise Covariance')
pp.xlabel('Node')
pp.ylabel('Node')

pp.subplot(133)
pp.imshow(Q0_fitted, cmap='viridis', aspect='auto')
pp.colorbar(label='Covariance')
pp.title('Steady-state Covariance')
pp.xlabel('Node')
pp.ylabel('Node')

pp.tight_layout()
pp.show()

# Plot convergence history if available
if 'correlation history' in mou_fitted.d_fit:
    pp.figure(figsize=(12, 4))
    
    pp.subplot(121)
    pp.plot(mou_fitted.d_fit['distance history'][:mou_fitted.d_fit['iterations']])
    pp.xlabel('Iteration')
    pp.ylabel('Distance')
    pp.title('Model Error During Optimization')
    pp.grid(True)
    
    pp.subplot(122)
    pp.plot(mou_fitted.d_fit['correlation history'][:mou_fitted.d_fit['iterations']])
    pp.xlabel('Iteration')
    pp.ylabel('Pearson correlation')
    pp.title('Model Fit During Optimization')
    pp.grid(True)
    
    pp.tight_layout()
    pp.show()


# %%
