import math
import paddle
import paddle.nn.functional as F

def npo2(length):
    """Next power of two >= length"""
    return 2 ** math.ceil(math.log2(length))

def pad_npo2(X):
    """
    Pad tensor along length dim (axis=1) to next power of two.
    X: (B, L, D, N)
    """
    L = X.shape[1]
    L2 = npo2(L)
    if L2 == L:
        return X
    pad_len = L2 - L
    pad_shape = [X.shape[0], pad_len, X.shape[2], X.shape[3]]
    pad_tensor = paddle.zeros(pad_shape, dtype=X.dtype)
    return paddle.concat([X, pad_tensor], axis=1)


class PScan(paddle.autograd.PyLayer):
    """
    Paddle port of the PyTorch PScan (Blelloch parallel scan) autograd.Function.

    Usage:
        pscan = PScan.apply
        out = pscan(A, X)
    where A, X have shape (B, L, D, N)
    """

    @staticmethod
    def _pscan_inplace(A, X):
        """
        Perform in-place Blelloch style parallel scan on tensors
        A : (B, D, L, N)
        X : (B, D, L, N)
        This function mutates X (and also mutates Aa/A views).
        The code follows the same structure as the PyTorch version.
        """
        B, D, L, _ = A.shape
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        # up-sweep (collapse pairs repeatedly)
        for _ in range(num_steps - 2):
            T = Xa.shape[2]
            # reshape to group pairs
            Aa = Aa.reshape([B, D, T // 2, 2, -1])
            Xa = Xa.reshape([B, D, T // 2, 2, -1])

            # Xa[...,1] += Aa[...,1] * Xa[...,0]
            Xa_slice_1 = Xa[:, :, :, 1]
            Xa_slice_0 = Xa[:, :, :, 0]
            Aa_slice_1 = Aa[:, :, :, 1]
            # do assignment (in-place semantics via slice assign)
            Xa[:, :, :, 1] = Xa_slice_1 + Aa_slice_1 * Xa_slice_0

            # Aa[...,1] *= Aa[...,0]
            Aa[:, :, :, 1] = Aa_slice_1 * Aa[:, :, :, 0]

            # reduce to the "second" nodes
            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        # handle last 4/2/1 nodes cases
        if Xa.shape[2] == 4:
            Xa[:, :, 1] = Xa[:, :, 1] + Aa[:, :, 1] * Xa[:, :, 0]
            Aa[:, :, 1] = Aa[:, :, 1] * Aa[:, :, 0]
            Xa[:, :, 3] = Xa[:, :, 3] + Aa[:, :, 3] * (Xa[:, :, 2] + Aa[:, :, 2] * Xa[:, :, 1])
        elif Xa.shape[2] == 2:
            Xa[:, :, 1] = Xa[:, :, 1] + Aa[:, :, 1] * Xa[:, :, 0]
            return
        else:
            return

        # down-sweep
        step = 2 ** (num_steps - 2)
        Aa = A[:, :, step - 1:L:step]
        Xa = X[:, :, step - 1:L:step]
        Xa[:, :, 2] = Xa[:, :, 2] + Aa[:, :, 2] * Xa[:, :, 1]
        Aa[:, :, 2] = Aa[:, :, 2] * Aa[:, :, 1]

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 2**k - 1:L:2**k]
            Xa = X[:, :, 2**k - 1:L:2**k]

            T = Xa.shape[2]
            Aa = Aa.reshape([B, D, T // 2, 2, -1])
            Xa = Xa.reshape([B, D, T // 2, 2, -1])

            # Xa[:, :, 1:, 0] += Aa[:, :, 1:, 0] * Xa[:, :, :-1, 1]
            left = Xa[:, :, 1:, 0]
            right = Xa[:, :, :-1, 1]
            mult = Aa[:, :, 1:, 0]
            Xa[:, :, 1:, 0] = left + mult * right

            # Aa[:, :, 1:, 0] *= Aa[:, :, :-1, 1]
            Aa[:, :, 1:, 0] = Aa[:, :, 1:, 0] * Aa[:, :, :-1, 1]

    @staticmethod
    def _pscan_rev_inplace(A, X):
        """
        Reverse-version of pscan used in backward (in-place on X).
        A : (B, D, L, N)
        X : (B, D, L, N)
        """
        B, D, L, _ = A.shape
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.shape[2]
            Aa = Aa.reshape([B, D, T // 2, 2, -1])
            Xa = Xa.reshape([B, D, T // 2, 2, -1])

            Xa[:, :, :, 0] = Xa[:, :, :, 0] + Aa[:, :, :, 0] * Xa[:, :, :, 1]
            Aa[:, :, :, 0] = Aa[:, :, :, 0] * Aa[:, :, :, 1]

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        if Xa.shape[2] == 4:
            Xa[:, :, 2] = Xa[:, :, 2] + Aa[:, :, 2] * Xa[:, :, 3]
            Aa[:, :, 2] = Aa[:, :, 2] * Aa[:, :, 3]
            Xa[:, :, 0] = Xa[:, :, 0] + Aa[:, :, 0] * (Xa[:, :, 1] + Aa[:, :, 1] * Xa[:, :, 2])
        elif Xa.shape[2] == 2:
            Xa[:, :, 0] = Xa[:, :, 0] + Aa[:, :, 0] * Xa[:, :, 1]
            return
        else:
            return

        Aa = A[:, :, 0:L:2**(num_steps - 2)]
        Xa = X[:, :, 0:L:2**(num_steps - 2)]
        Xa[:, :, 1] = Xa[:, :, 1] + Aa[:, :, 1] * Xa[:, :, 2]
        Aa[:, :, 1] = Aa[:, :, 1] * Aa[:, :, 2]

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 0:L:2**k]
            Xa = X[:, :, 0:L:2**k]

            T = Xa.shape[2]
            Aa = Aa.reshape([B, D, T // 2, 2, -1])
            Xa = Xa.reshape([B, D, T // 2, 2, -1])

            Xa[:, :, :-1, 1] = Xa[:, :, :-1, 1] + Aa[:, :, :-1, 1] * Xa[:, :, 1:, 0]
            Aa[:, :, :-1, 1] = Aa[:, :, :-1, 1] * Aa[:, :, 1:, 0]

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        A_in : (B, L, D, N)
        X_in : (B, L, D, N)
        returns H : (B, L, D, N)
        """
        L = X_in.shape[1]

        # clone/pad (cloning needed because we'll do in-place ops on X)
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            A = pad_npo2(A_in).clone()
            X = pad_npo2(X_in).clone()

        # transpose to (B, D, L, N)
        A = paddle.transpose(A, [0, 2, 1, 3])
        X = paddle.transpose(X, [0, 2, 1, 3])

        # perform in-place parallel scan on X
        PScan._pscan_inplace(A, X)

        # save original A_in and scanned X (X is mutated to be the scan result)
        ctx.save_for_backward(A_in, X)

        # transpose back to (B, L, D, N) and slice to original length
        out = paddle.transpose(X, [0, 2, 1, 3])[:, :L]
        return out

    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Returns gradA, gradX (both same shapes as A_in, X_in)
        """
        # retrieve saved tensors
        A_in, X_scanned = ctx.saved_tensor()  # A_in: (B, L, D, N), X_scanned: (B, D, L, N) saved as in forward

        L = grad_output_in.shape[1]

        # pad if necessary
        if L == npo2(L):
            grad_output = grad_output_in.clone()
        else:
            grad_output = pad_npo2(grad_output_in)

            # also pad A_in so sizes align (we will transpose it)
            A_in = pad_npo2(A_in)

        # transpose to (B, D, L, N)
        grad_output = paddle.transpose(grad_output, [0, 2, 1, 3])
        A_in_t = paddle.transpose(A_in, [0, 2, 1, 3])  # (B, D, L, N)

        # create A shifted-left: A = pad(A_in_t[:, :, 1:], pad_last=1)
        # equivalent to torch.nn.functional.pad(A_in_t[:, :, 1:], (0,0,0,1))
        B_, D_, Lpad, N_ = A_in_t.shape
        # slice from idx 1 to end, then append zeros at last time position
        A_shift = A_in_t[:, :, 1:, :]
        zero_tail = paddle.zeros([B_, D_, 1, N_], dtype=A_in_t.dtype)
        A = paddle.concat([A_shift, zero_tail], axis=2)  # (B, D, L, N)

        # reverse parallel scan on grad_output (in-place)
        PScan._pscan_rev_inplace(A, grad_output)

        # compute Q (grad wrt A)
        # X_scanned is saved in forward as transposed to (B,D,L,N)
        X = X_scanned  # (B, D, L, N)
        Q = paddle.zeros_like(X)
        # grad_A_inter = X[:, :, :-1] * grad_output[:, :, 1:]
        inter = X[:, :, :-1, :] * grad_output[:, :, 1:, :]
        inter = paddle.nan_to_num(inter)
        # Q[:, :, 1:] += inter
        Q[:, :, 1:, :] = Q[:, :, 1:, :] + inter

        # transpose gradients back to original shape (B, L, D, N) and slice to original L
        gradA = paddle.transpose(Q, [0, 2, 1, 3])[:, :L]
        gradX = paddle.transpose(grad_output, [0, 2, 1, 3])[:, :L]
        return gradA, gradX


# alias like in pytorch
pscan = PScan.apply
