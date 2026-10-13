import sys
import logging

import scipy
import quaternion
import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


def affine_ref_to_tar(ref, tar):
    """ Get homogenous transform to transform ref to tar using an Affine matrix

    ref and tar are npArrays of shape (3, n)

    Affine matrix allows for Translation, Rotation, non-uniform scaling,
    and shearing
    """

    if ref.shape[0] == 3:
        ref = ref.T  # lstsq solver needs them in (n, 3) but want them to be
        tar = tar.T  # consistent with horn's algorithm
    elif ref.shape[1] == 3:
        logger.info('ref and tar input with shape (n, 3)')
    else:
        logger.error('ref and tar should have shape (3, n)')
        return None

    if np.shape(ref) != np.shape(tar):
        logger.error('Shape of ref %s and tar %s must be the same'
                     % (np.shape(ref), np.shape(tar)))

    X = pad_column(ref)  # Add column of ones to make (N, 4)
    Y = pad_column(tar)

    H, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)  # Y = X * H

    H[np.abs(H) < 1e-10] = 0  # set really small values to zero
    H = H.T  # So it can be left multiplied by column vector points (4, N)

    return H


def fit_ref_to_tar(ref, tar, translation=True, rotation=True, scale=False):
    """ Get homogeneous transform for fitting ref to tar using Horn's Method

    ref and tar are npArrays of shape (3, n)

    Horn's method allows for translation, rotation, and uniform scaling

    Return H
    """

    if np.shape(ref) != np.shape(tar):
        logger.error("Shape of ref %s and tar %s must be the same"
                     % (np.shape(ref), np.shape(tar)))
        sys.exit()

    r, c = np.shape(ref)

    if r != 3:
        logger.error("ref and tar had %i rows. Expected 3" % r)
        sys.exit()

    # Center the input data
    lc = np.zeros(r)  # Default when translation is not fitted
    rc = np.zeros(r)
    if translation:
        lc = np.mean(ref, axis=1)  # Array of shape (d,)
        rc = np.mean(tar, axis=1)

    left = ref - lc[:, np.newaxis]  # Array of shape (d, n)
    right = tar - rc[:, np.newaxis]

    R = np.eye(3)  # Default when rotation is not fitted
    if rotation:
        # M = [[ Sxx, Sxy, Sxz],
        #      [ Syx, Syy, Syz],
        #      [ Szx, Zzy, Szz]]
        M = left @ right.T  # Array of shape (d, d)
        [Sxx, Syx, Szx, Sxy, Syy, Szy, Sxz, Syz, Szz] = M.T.flatten()

        # N is a symmetric 4x4 matrix
        N = np.array([[Sxx+Syy+Szz,   Syz - Szy,   Szx - Sxz,   Sxy - Syx],
                      [Syz - Szy, Sxx-Syy-Szz,   Sxy + Syx,   Szx + Sxz],
                      [Szx - Sxz,   Sxy + Syx, -Sxx+Syy-Szz,   Syz + Szy],
                      [Sxy - Syx,   Szx + Sxz,   Syz + Szy, -Sxx-Syy+Szz]])

        # Eigenvalues and eigenvectors
        [D, V] = np.linalg.eig(N)

        maxEig = np.where(D == np.max(D))

        # Quaternion of the rotation is the principal eigenvector
        q = V[:, maxEig].flatten()

        # Compute rotation matrix
        v = V[1:, maxEig].flatten()
        Z = np.array([[q[0], -q[3], q[2]],
                      [q[3], q[0], -q[1]],
                      [-q[2], q[1], q[0]]])
        R = v[:, np.newaxis] @ v[np.newaxis, :] + Z @ Z

    scaleFactor = 1  # Default if scale is not fitted
    if scale:
        scaleFactor = np.sum(right * (R @ left)) / np.sum(left**2)

    # Compute translation matrix
    t = rc[:, np.newaxis] - R @ (lc[:, np.newaxis] * scaleFactor)

    # Homogenous Transformation matrix
    H = np.eye(4)
    H[:3, :3] = R*scaleFactor
    H[:3, -1] = t.flatten()

    return H


def fit_ref_to_tar_lc(ref, tar, translation, rotation, scale):
    """ Get homogeneous transform for fitting ref to tar using Horn's Method

    ref and tar are npArrays of shape (3, n)

    Horn's method allows for translation, rotation, and uniform scaling

    Return H
    """

    if np.shape(ref) != np.shape(tar):
        logger.error("Shape of ref %s and tar %s must be the same"
                     % (np.shape(ref), np.shape(tar)))
        sys.exit()

    r, c = np.shape(ref)

    if r != 3:
        logger.error("ref and tar had %i rows. Expected 3" % r)
        sys.exit()

    # Center the input data
    lc = np.zeros(r)  # Default when translation is not fitted
    rc = np.zeros(r)
    if translation:
        lc = np.mean(ref, axis=1)  # Array of shape (d,)
        rc = np.mean(tar, axis=1)

    left = ref - lc[:, np.newaxis]  # Array of shape (d, n)
    right = tar - rc[:, np.newaxis]

    R = np.eye(3)  # Default when rotation is not fitted
    if rotation:
        # M = [[ Sxx, Sxy, Sxz],
        #      [ Syx, Syy, Syz],
        #      [ Szx, Zzy, Szz]]
        M = left @ right.T  # Array of shape (d, d)
        [Sxx, Syx, Szx, Sxy, Syy, Szy, Sxz, Syz, Szz] = M.T.flatten()

        # N is a symmetric 4x4 matrix
        N = np.array([[Sxx+Syy+Szz,   Syz - Szy,   Szx - Sxz,   Sxy - Syx],
                      [Syz - Szy, Sxx-Syy-Szz,   Sxy + Syx,   Szx + Sxz],
                      [Szx - Sxz,   Sxy + Syx, -Sxx+Syy-Szz,   Syz + Szy],
                      [Sxy - Syx,   Szx + Sxz,   Syz + Szy, -Sxx-Syy+Szz]])

        # Eigenvalues and eigenvectors
        [D, V] = np.linalg.eig(N)

        maxEig = np.where(D == np.max(D))

        # Quaternion of the rotation is the principal eigenvector
        q = V[:, maxEig].flatten()

        # Compute rotation matrix
        v = V[1:, maxEig].flatten()
        Z = np.array([[q[0], -q[3], q[2]],
                      [q[3], q[0], -q[1]],
                      [-q[2], q[1], q[0]]])
        R = v[:, np.newaxis] @ v[np.newaxis, :] + Z @ Z

    scaleFactor = 1  # Default if scale is not fitted
    if scale:
        scaleFactor = np.sum(right * (R @ left)) / np.sum(left**2)

    # Compute translation matrix
    t = rc[:, np.newaxis] - R @ (lc[:, np.newaxis] * scaleFactor)

    # Homogenous Transformation matrix
    H = np.eye(4)
    H[:3, :3] = R*scaleFactor
    H[:3, -1] = t.flatten()

    return H

def compute6Dframe(rom, n3d):
    """ Computes the 6D transform using Horn's Method for a single frame of data

    This is similar to fit_ref_to_tar() but removes additional options

    rom (nparray): marker positions with shape (3, n) where n is the # markers
    n3d (nparray): 3d marker data with shape(3, n)

    TODO:
        Add 3D RMS and individual marker error metrics
    """
    # Handle bad floats
    n3d = np.where(n3d < -3e20, np.nan, n3d)

    # Mask missing data
    mask = np.isnan(n3d).any(axis=0)
    rom = rom[:, ~mask]
    n3d = n3d[:, ~mask]

    r, c = np.shape(rom)

    if c < 3:
        return np.ones(7)*np.nan

    # Center the input data
    lc = np.mean(rom, axis=1)  # Array of shape (3,)
    rc = np.mean(n3d, axis=1)

    left = rom - lc[:, np.newaxis]  # Array of shape (3, n)
    right = n3d - rc[:, np.newaxis]

    # M = [[ Sxx, Sxy, Sxz],
    #      [ Syx, Syy, Syz],
    #      [ Szx, Zzy, Szz]]
    M = left @ right.T  # Array of shape (3, 3)
    [Sxx, Syx, Szx, Sxy, Syy, Szy, Sxz, Syz, Szz] = M.T.flatten()

    # N is a symmetric 4x4 matrix
    N = np.array([[Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
                  [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
                  [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
                  [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz]])

    # Eigenvalues and eigenvectors
    [D, V] = np.linalg.eig(N)
    maxEig = np.where(D == np.max(D))

    # Quaternion of the rotation is the principal eigenvector
    q = quaternion.quaternion(*V[:, maxEig].flatten())
    q = q * np.sign(q.w)

    # Compute translation matrix
    t = rc - quaternion.rotate_vectors(q, lc)
    out = np.hstack([q.components, t])
    return out


def apply_h_transform(ref, H):
    """ Use homogeneous transform to convert ref to coordinate system of tar
    ref is a np array of shape (3, n)
    """

    r = ref.shape[0]
    if r != 3:
        logger.error('ref has %i rows. Expected 3.' %r)
        sys.exit()

    # Convert to (4, n) for matrix multiplcation
    ref4 = pad_row(ref)
    ref_transformed = unpad_row(H @ ref4)
    return ref_transformed


def get_resid(ref_transformed, tar):
    """ Calculate residual between Transformed ref and tar """
    return np.sum((ref_transformed - tar)**2, axis=0)**0.5


def get_rmse(resid):
    """ Return RMSE of residuals """
    return np.mean(resid**2)**0.5


def decompose(H):
    """ Return translation, rotation, scale, and shear matrices

    From answer on stackexchange by aerobiomat
    https://math.stackexchange.com/questions/237369/given-this-transformation-matrix-how-do-i-decompose-it-into-translation-rotati/417813
    """
    translation = H[:-1, -1]
    M = H[:3, :3]

    # Polar decomposition into Rotation and Scale + Shear matricies
    R, S = scipy.linalg.polar(M)

    # Handle case where rotation is a reflection
    sign = np.sign(np.linalg.det(M))
    R *= sign
    S *= sign

    rotation = quaternion.from_rotation_matrix(R)
    rotation *= np.sign(rotation.w)

    S[S < 1e-10] = 0  # Clean up very small numbers

    scale = np.diag(S)
    shear = np.array(S)  # fill diagonal is performed inplace
    np.fill_diagonal(shear, 1)

    return translation, rotation, scale, shear


def compose(translation=None, rotation=None, scale=None, shear=None):
    """ Create a homogeneous transformation matrix from components

    translation should be a [3, 1]
    rotation should be quaternion
    scale should be [3, 1] or scalar
    shear should be [3, 3]
    """

    H = np.eye(4)
    R = np.eye(3)
    S = np.eye(3)

    if shear is not None:
        S = shear

    if scale is not None:
        np.fill_diagonal(S, scale)

    if rotation is not None:
        R = quaternion.as_rotation_matrix(rotation)

    if translation is not None:
        H[:-1, -1] = translation

    L = R @ S
    H[:3, :3] = L
    return H


def get_axis_angle(q):
    """ Return [axis, angle] from quaterion """

    angle = q.angle()  # np.arcsin(np.linalg.norm(q.vec))*2
    if angle:
        axis = quaternion.as_rotation_vector(q) / angle
    else:
        axis = [np.nan, np.nan, np.nan]
    return [axis, angle]


def pad_column(x):
    """ Add column of ones to matrix x"""
    return np.hstack([x, np.ones((x.shape[0], 1))])


def pad_row(x):
    """ Add row of ones to matrix x"""
    return np.vstack([x, np.ones((x.shape[1]))])


def unpad_row(x):
    """ Drop last row from matrix x"""
    return x[:-1, :]


def unpad_col(x):
    """ Drop last column from matrix x"""
    return x[:, :-1]
    
def inverse_H(H):

    inv_H = H.copy()
    translation = H[:3, 3].copy()
    R = H[:3, :3].copy()
    #print(translation)
    #print(R.T @ translation)
    
    inv_H[:-1, -1] = -1 * (R.T @ translation)
    inv_H[:3, :3] = H[:3, :3].T
    
    return inv_H

def euler_to_H(input):
    #input has shape np.array([Rz, Ry, Rx, Tx, Ty, Tz])
    angles = input[:3]
    translation = input[3:]
    r = Rotation.from_euler("ZYX", angles, degrees=True)
    M = r.as_matrix()
    H = np.zeros((4, 4))
    H[:3, :3] = M
    H[:3, 3] = translation
    H[3, 3] = 1
    return H
    
def H_to_euler(input):
    #input 4x4 matrix
    M = input[:3, :3]
    translation = input[:3, 3]
    r = Rotation.from_matrix(M)
    angles = r.as_euler("ZYX", degrees=True)

    pose = np.zeros(6)
    pose[:3] = angles
    pose[3:] = translation
    return pose