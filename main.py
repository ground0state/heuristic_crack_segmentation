import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hessian_matrix


def read_img(filename):
    img = cv2.imread(filename)
    if img is None:
        raise FileNotFoundError()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def image_correction(img):
    """濃淡差や影を補正する."""
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def norm2d(x, y, sigma):
    Z = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return Z


def gaussian_kernel(size, sigma=None):
    if size % 2 == 0:
        print('kernel size should be odd')
        return

    if sigma is None:
        sigma = (size - 1) / 2

    # [0,size]→[-sigma, sigma] にずらす
    x = y = np.arange(0, size) - (size - 1) / 2
    X, Y = np.meshgrid(x, y)

    mat = norm2d(X, Y, sigma)

    # 総和が1になるように
    kernel = mat / np.sum(mat)
    return kernel


def norm2d_diff(x, y, sigma, direction='xx'):
    if direction == 'xx':
        Z = (x**2 - sigma**2) * np.exp(-(x**2 + y**2) /
                                       (2 * sigma**2)) / (2 * np.pi * sigma**6)
    elif direction == 'yy':
        Z = (y**2 - sigma**2) * np.exp(-(x**2 + y**2) /
                                       (2 * sigma**2)) / (2 * np.pi * sigma**6)
    elif direction == 'xy':
        Z = x * y * np.exp(-(x**2 + y**2) /
                           (2 * sigma**2)) / (2 * np.pi * sigma**6)
    else:
        raise ValueError("Direction is invalid.")
    return Z


def hessian_kernel(size, sigma=None, direction='xx'):
    if size % 2 == 0:
        print('kernel size should be odd')
        return

    if sigma is None:
        sigma = (size - 1) / 2

    # [0,size]→[-sigma, sigma] にずらす
    x = y = np.arange(0, size) - (size - 1) / 2
    X, Y = np.meshgrid(x, y)

    kernel = norm2d_diff(X, Y, sigma, direction)

    # 総和が1になるように
    if direction == 'xx' or direction == 'yy':
        kernel = kernel / np.sum(kernel)
    return kernel


def make_image_hessian(img, sigma):
    Hxx, Hxy, Hyy = hessian_matrix(
        img, sigma=sigma, mode='mirror', order='xy')
    hessian = np.stack([Hxx, Hxy, Hxy, Hyy],
                       axis=2).reshape(*img.shape, 2, 2)
    return hessian


def hessian_emphasis(img, sigma=1.4, alpha=0.25):
    hessian = make_image_hessian(img, sigma)
    eig_val, _ = np.linalg.eig(hessian)

    r = np.zeros_like(img, dtype=np.float64)
    for i in range(hessian.shape[0]):
        for j in range(hessian.shape[1]):
            # 各画素の固有値を取得
            lambda1, lambda2 = eig_val[i, j]

            # 小さい方をlambda2にする
            if lambda2 > lambda1:
                temp = lambda1
                lambda1 = lambda2
                lambda2 = temp
            if lambda1 <= 0:
                # 粒状構造
                lambda12 = abs(lambda2) + lambda1
            elif lambda2 < 0 < lambda1 < abs(lambda2) / alpha:
                # 線状構造
                lambda12 = abs(lambda2) - lambda1 * alpha
            else:
                lambda12 = 0
            r[i, j] = sigma**2 * lambda12
    return r


def multiscale_hessian_emphasis(
        img,
        sigma=1.4,
        alpha=0.25,
        s=1.4,
        num_iteration=4
):
    R = []
    for i in range(num_iteration):
        r = hessian_emphasis(img, sigma=sigma * s**i, alpha=alpha)
        R.append(r)
    R = np.stack(R, axis=0)

    out = np.max(R, axis=0)
    out_scale = np.argmax(R, axis=0) + 1  # scaleを1から始める
    return out, out_scale


def stochastic_relaxation_method(img, alpha=1.0):
    """確率的弛緩法.

    Parameters
    ----------
    img : ndarray
        画素値は0-1に規格化.
        ひびの方が背景よりも輝度が高い.

    Returns
    -------
    out
        確率的弛緩法を適用後の画像.
    residual
        確率的弛緩法適用前後の2乗誤差.
    """
    P_c = img / np.max(img)
    P_b = 1 - P_c

    # kernels = [
    #     np.array([[0, 0.5, 0], [0, 0, 0], [0, 0.5, 0]]),
    #     np.array([[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]]),
    #     np.array([[0, 0, 0], [0.5, 0, 0.5], [0, 0, 0]]),
    #     np.array([[0.5, 0, 0], [0, 0, 0], [0, 0, 0.5]])
    # ]

    kernels = [
        np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]) / 3,
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]) / 3,
        np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]) / 3,
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) / 3
    ]

    Q_c = []
    for kernel in kernels:
        q = cv2.filter2D(P_c, -1, kernel, borderType=cv2.BORDER_DEFAULT)
        Q_c.append(q)
    Q_c = np.stack(Q_c, axis=0)
    Q_b = 1 - Q_c

    P_c_new = np.divide(
        alpha * P_c[None, ...] * Q_c,
        alpha * P_c[None, ...] * Q_c + P_b[None, ...] * Q_b,
        where=P_c[None, ...] * Q_c + P_b[None, ...] * Q_b != 0)

    out = np.max(P_c_new, axis=0)
    return out


def find_threshold(img):
    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
        q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1, b2 = np.hsplit(bins, [i])  # weights
        # finding means and variances
        m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
        v1, v2 = np.sum(((b1 - m1)**2) * p1) / q1, np.sum(((b2 - m2)**2) * p2) / q2
        # calculates the minimization function
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    return thresh


def take_log(img):
    return np.log1p(img)


def stepwise_threshold_processing(
        img, candidate,
        iter_stp=50,
        k_size=11,
        element=None):

    width, height = img.shape
    if element is None:
        element = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)

    completed = set()
    t = b = r = l = int((k_size - 1) * 0.5)
    for _ in range(iter_stp):
        flag_updated = False
        # 候補領域の外周の位置インデックスを得る
        dilated = cv2.dilate(candidate, element, iterations=1)
        outline = dilated - candidate
        xs, ys = (outline == 1).nonzero()

        for x, y in zip(xs, ys):
            if (x, y) in completed:
                # 判定済み領域は除く
                continue
            # 注目画素近傍で大津の二値化のしきい値を求める
            xmin = max(x - l, 0)
            xmax = min(x + r + 1, width)
            ymin = max(x - t, 0)
            ymax = min(x + b + 1, height)
            patch = img[xmin:xmax, ymin:ymax]
            patch_candidate = candidate[xmin:xmax, ymin:ymax]

            if crack_judge(img, (x, y), patch, patch_candidate):
                # しきい値以上であれば新たな候補領域にする
                candidate[x, y] = 1.0
                flag_updated = True
            else:
                # しきい値未満で背景に確定する
                completed.add((x, y))

        if not flag_updated:
            break
    return candidate


def trans2uint(img):
    img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype('uint8')
    return img


def extract_crack_mean(patch, patch_candidate):
    mu_c = np.mean(patch[patch_candidate == 1])
    mu_b = np.mean(patch[patch_candidate == 0])
    return mu_c, mu_b


def crack_judge(img, coor, patch, patch_candidate, beta=0.9):
    x, y = coor
    mu_c, mu_b = extract_crack_mean(patch, patch_candidate)
    res = (abs(img[x, y] - mu_c) / abs(img[x, y] - mu_b + 1.e-9) * beta) <= 1

    # もう一つの方法
    # thresh = find_threshold(patch)
    # res = img[x, y] >= thresh
    return res


def crack_segmentation(
        img,
        iter_mhe=4,
        iter_srm=10,
        iter_stp=50):

    # 画像の前処理
    img = image_correction(img)
    # ヘッシアンの固有値別に画像を補正
    img, _ = multiscale_hessian_emphasis(
        img,
        sigma=1.4,
        alpha=0.25,
        s=1.4,
        num_iteration=iter_mhe
    )

    ori_img = img.copy()
    ori_img = trans2uint(ori_img)

    # 画素値をスケール補正
    img = take_log(img)

    # 確率的弛緩法でノイズを消す
    for _ in range(iter_srm):
        img = stochastic_relaxation_method(img, alpha=1.4)

    # 抽出候補を二値画像として得る
    # img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype('uint8')
    # _, candidate = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # candidate = candidate / 255

    _, candidate = cv2.threshold(img, .5, 1., cv2.THRESH_BINARY)

    # 消しすぎた候補領域を段階的閾値処理で拡張する
    mask = stepwise_threshold_processing(
        ori_img, candidate, iter_stp=iter_stp, k_size=27)
    return mask


def main():
    img = read_img('sample01.jpg')
    img = cv2.bitwise_not(img)

    plt.imshow(img, 'gray')
    plt.show()

    mask = crack_segmentation(
        img,
        iter_mhe=4,
        iter_srm=10,
        iter_stp=80)

    plt.imshow(mask, 'gray')
    plt.show()


if __name__ == '__main__':
    main()
