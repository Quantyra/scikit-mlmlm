"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from warnings import showwarning
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist
import sys


class MultiLabelMLMClassifier(MultiOutputClassifier):
    def __init__(self, p_grid):
        self.p_grid = p_grid

    def fit(self, X, y):
        # Check that X and y have correct shape
        # X, y = check_X_y(X, y)
        # Store the classes seen during fit
        # self.classes_ = unique_labels(y)

        p_grid = self.p_grid

        # Distance regression model training with Moore-Penrose pseudoinverse.
        [Dx, Irem, Ik, R, K, dx_alpha, Dy, P, hat_matrix, B_p] = self._dist_reg_train(
            X, y
        )

        # Compute Local Rcut thresholding values
        Klr = self._local_rcut(y, pairwise_distances(X, R) @ B_p)

        p_best, tc, score = self._loocv_train(hat_matrix, Dy, X, y, p_grid)

        self.score = score
        self.p_best = p_best
        self.tc = tc
        self.Klr = Klr
        self.Dx = Dx
        self.Irem = Irem
        self.Ik = Ik
        self.R = R
        self.K = K
        self.dx_alpha = dx_alpha
        self.Dy = Dy
        self.P = P
        self.hat_matrix = hat_matrix
        self.B_p = B_p
        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        """A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ["X_", "y_"])

        # Input validation
        X = check_array(X)

        y = self._ml_mlm_pred(X)
        y_pred = y > self.tc
        return y_pred

    def predict_proba(self, X):
        """A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ["X_", "y_"])

        # Input validation
        X = check_array(X)

        y = self._ml_mlm_pred(X)
        return y

    def _loocv_train(self, hat_matrix, Dy, Xtrain, Ytrain, p_grid):
        # ML-MLM training with LOOCV using PRESS ranking loss statistic
        Yhat = hat_matrix @ Dy
        N = Xtrain.shape[0]

        # Out-of-sample distance estimates via LOOCV
        Dy_pred_PRESS = np.abs(
            (Yhat - np.diag(hat_matrix) * Dy) / (np.ones(N) - np.diag(hat_matrix))
        )

        score = np.zeros(p_grid.shape[0])
        rloss_best = sys.float_info.max
        for p_ind in range(p_grid.shape[0]):
            p = p_grid[p_ind]
            Rho_inv = 1 / np.power(Dy_pred_PRESS, p)  # Rho_inv is K x N
            r_sum = np.sum(Rho_inv, axis=1)  # r_sum is N x 1
            maxrho = np.max(Rho_inv, axis=1)  # maxrho is N x 1

            ## Inf handling for larger power parameter values
            if np.any(np.isinf(maxrho)):
                Im = np.where(maxrho == np.inf)[0]
                print(f"Handling close to zero distances for {len(Im)} cases")
                for ii in range(len(Im)):
                    Rho_inv[Im[ii], Rho_inv[Im[ii], :] == np.inf] = 1
                    Rho_inv[Im[ii], Rho_inv[Im[ii], :] != np.inf] = 0
                    r_sum[Im[ii]] = np.sum(Rho_inv[Im[ii], :])

            Irs = np.where(r_sum == np.inf)[0]
            if len(Irs) > 0:
                print(f"Handling inf divisor for {len(Irs)} cases")
                r_sum[Irs] = maxrho[Irs]

            W = Rho_inv / r_sum[:, np.newaxis]
            Yscore = W @ Ytrain
            loss_score, _ = self._ranking_loss(Ytrain, Yscore)
            score[p_ind] = loss_score
            print(f"p = {p}, rloss statistic =  {score[p_ind]}")
            if score[p_ind] < rloss_best:
                rloss_best = score[p_ind]
                Yscore_temp = Yscore

        # Select an optimized ML-MLM model according to the smallest LOOCV ranking loss statistic
        min_p_ind = np.argmin(score)
        p_best = p_grid[min_p_ind]

        # Cardinality-based thresholding selection for the optimized ML-MLM model
        CtrN = int(np.sum(Ytrain))
        Ytc = sorted(Yscore_temp.flatten(), reverse=True)
        tc = (Ytc[CtrN - 1] + Ytc[CtrN]) / 2

        return p_best, tc, score

    def _dist_reg_train(self, Xtrain, Ytrain):
        # Computing input space distance matrix
        Dx = pairwise_distances(Xtrain, Xtrain)
        Irem, Ik = self._find_rem_inds(Xtrain)
        Dx = np.delete(Dx, Irem, axis=1)
        R = Xtrain
        R = np.delete(R, Irem, axis=0)
        K = R.shape[0]
        dx_alpha = np.quantile(pdist(R), 0.001)

        # Computing output space distance matrix
        Dy = pairwise_distances(Ytrain, Ytrain)

        # Computing pinv
        P = np.linalg.pinv(Dx.T @ Dx + dx_alpha * np.eye(K))

        # Computing hat matrix
        hat_matrix = Dx @ P @ Dx.T

        # Solving distance regression model coefficients
        B_p = P @ Dx.T @ Dy

        return Dx, Irem, Ik, R, K, dx_alpha, Dy, P, hat_matrix, B_p

    def _ranking_loss(self, Ygt, Ypred):
        N, M = Ygt.shape
        score = 0
        score_arr = np.zeros(N)
        Nd = N
        for ii in range(N):
            Lii = np.sum(Ygt[ii])
            if Lii > 0:
                inds = np.argsort(Ypred[ii])[::-1]
                ytemp = Ygt[ii, inds]
                ytemp_search = ~ytemp
                rankloss_temp = 0
                while np.sum(ytemp_search) > 0:
                    ind_temp = np.where(ytemp_search)[0][0]
                    rankloss_temp += np.sum(ytemp[ind_temp:])
                    ytemp = ytemp[ind_temp + 1 :]
                    ytemp_search = ~ytemp
                rankloss_temp /= Lii * (M - Lii)
                score_arr[ii] = rankloss_temp
                score += rankloss_temp
            else:
                Nd -= 1
        score /= Nd
        return score, score_arr

    def _ml_mlm_pred(self, Xtest):
        R = self.R
        B = self.B_p
        p = self.p_best
        N = Xtest.shape[0]
        print("Computing prediction for a test set N =", N)

        # Predict distances in label space
        pred_dists = np.dot(pairwise_distances(Xtest, R), B)

        # Prepare inverse distance weighting components
        Rho_inv = 1 / np.abs(pred_dists) ** p  # Rho_inv is N x K
        r_sum = np.sum(Rho_inv, axis=1)  # r_sum is K x 1
        maxrho = np.max(Rho_inv, axis=1)  # maxrho is K x 1

        # Inf handling for larger power parameter values (typically when  p ~ 100)
        if np.max(maxrho) == np.inf:
            Im = np.where(maxrho == np.inf)[0]
            print("Handling close to zero distances for", len(Im), "cases")
            for ii in Im:
                Rho_inv[ii, Rho_inv[ii, :] == np.inf] = 1
                Rho_inv[ii, Rho_inv[ii, :] != np.inf] = 0
                r_sum[ii] = np.sum(Rho_inv[ii, :])

        Irs = np.where(r_sum == np.inf)[0]
        if len(Irs) > 0:
            print("Handling inf divisor for", len(Irs), "cases")
            r_sum[Irs] = maxrho[Irs]

        # Scale the weights
        W = Rho_inv / r_sum[:, np.newaxis]

        # Construct convex combinations of label vectors (label scoring)
        Yscore = W @ self.y_

        return Yscore

    def _find_rem_inds(self, X):
        N = X.shape[0]
        _, I, _ = np.unique(X, axis=0, return_index=True, return_inverse=True)
        I = np.sort(I)
        Idp = np.setdiff1d(np.arange(N), I)

        return Idp, I

    def _local_rcut(self, T, pred_dists):
        min_inds = np.argmin(pred_dists, axis=1)
        Ts = T[min_inds]
        Klr = np.sum(Ts, axis=1)

        return Klr
