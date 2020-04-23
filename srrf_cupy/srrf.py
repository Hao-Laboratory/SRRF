#! /usr/bin/env python
# -- coding: utf-8 --
import argparse
import warnings
import os, sys, time
from math import ceil

import cupy as cp
import numpy as np
from matplotlib import animation as animation
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image, ImageSequence
from scipy import interpolate, optimize


# from mpl_toolkits.mplot3d import Axes3D  # axes3d，3d绘图


def get_kernel_path(kernel_filename):  # For compatibility of PyInstaller
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, kernel_filename)
    else:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), kernel_filename)


def read_cu_code(code_filename, params):
    with open(code_filename, "r", encoding="utf8") as f:
        code = f.read()
    for k, v in params.items():
        if isinstance(v, str):
            code = "#define " + k + " " + v + "\n" + code  # str object will be added directly
        else:
            code = "#define " + k + " " + str(v) + "\n" + code
    return code


def split_image_sequence(images: cp.array, batch_size: int) -> cp.array:
    """
    Generator
    Split image sequence to small sequences. Each of them will contain avg_every_n*batch_size frames
    Then the generator will yield these small sequences
    If all sequences have been yielded, the generator will yield a None.
    """
    frame_n = images.shape[0]
    len_images = int(frame_n / batch_size)
    if len_images is 0:
        yield images
    else:
        images = np.split(images[:len_images * batch_size], len_images)
        for image_mat in images:
            yield image_mat
    yield None


def avg_every_n_frames(image_mat: cp.array, avg_every_n: int) -> cp.array:
    h, w = image_mat.shape[1:3]
    mat = image_mat.reshape((avg_every_n, -1, h, w)).astype(cp.float32)
    mat = cp.average(mat, axis=0)
    return mat


def generate_lanczos4_weights_lut() -> cp.array:
    """generate a look-up table of lanczos4 core function for further acceleration.
    lanczos4 stands for 4-order lanczos(四阶lanczos算法，常用于图像放大,缩小一般只用到3阶）
    the LUT provides an accuracy of 1/(2^8). since the order is 4, the range of x is from[-4,4]
    this LUT is about 3 times faster than function evaluation counterparts.
    func_lut[abs((x*1024).astype(int))] == lanczos4_core_function(x)
    """
    # 我们需要的是core_lut，即图片上任何一个非整数坐标，从周围采样点插值时，各采样点的权重的查找表
    # 它的下标(index)取值范围为[-3,5), 使用core_lut[(x*1024).astype(int)]来查表
    i = cp.hstack((cp.arange(0, 5, 1 / 1024, dtype=cp.float32),
                   cp.arange(-3, 0, 1 / 1024, dtype=cp.float32)))  # this is the index of lut
    x = cp.zeros((len(i), 8), dtype=cp.float32)
    x -= cp.expand_dims(cp.arange(-3, 5), 0)
    x += cp.expand_dims(i, 1)
    core_lut = cp.sinc(x) * cp.sinc(x / 4)
    # normalize weights
    y = cp.mean(core_lut, axis=1, keepdims=True)
    return core_lut / y


def catmullrom_zoom_wrapper(images: cp.array,
                            mag: float,
                            frame_n: int,
                            ) -> cp.array:
    """ Zoom image using catmull_rom algorithm """

    coeff = 0.5 * cp.array([[0, 2, 0, 0], [-1, 0, 1, 0], [2, -5, 4, -1], [-1, 3, -3, 1]], dtype=cp.float32).T
    h, w = images.shape[1:3]
    # First interpolate in Y direction
    yCoordinate = cp.linspace(0, h - 1 / mag, int(h * mag), dtype=cp.float32)
    xCoordinate = cp.linspace(0, w - 1 / mag, int(w * mag), dtype=cp.float32)

    def generate_q_u_matrix(x_coordinate: cp.array, y_coordinate: cp.array) -> tuple:
        flatten_flag = x_coordinate.ndim > 1
        if flatten_flag:
            x_coordinate = x_coordinate.flatten()
            y_coordinate = y_coordinate.flatten()

        t, u = cp.modf(y_coordinate)
        u = u.astype(int)
        uy = cp.vstack([
            cp.minimum(cp.maximum(u - 1, 0), h - 1),
            cp.minimum(cp.maximum(u, 0), h - 1),
            cp.minimum(cp.maximum(u + 1, 0), h - 1),
            cp.minimum(cp.maximum(u + 2, 0), h - 1),
        ]).astype(int)
        Qy = cp.dot(coeff, cp.vstack([cp.ones_like(t, dtype=cp.float32), t, cp.power(t, 2), cp.power(t, 3)]))
        t, u = cp.modf(x_coordinate)
        u = u.astype(int)
        ux = cp.vstack([
            cp.minimum(cp.maximum(u - 1, 0), w - 1),
            cp.minimum(cp.maximum(u, 0), w - 1),
            cp.minimum(cp.maximum(u + 1, 0), w - 1),
            cp.minimum(cp.maximum(u + 2, 0), w - 1),
        ])
        Qx = cp.dot(coeff, cp.vstack([cp.ones_like(t, dtype=cp.float32), t, cp.power(t, 2), cp.power(t, 3)]))

        if flatten_flag:
            Qx = Qx.reshape(4, frame_n, int(w * mag)).transpose(1, 0, 2).copy()
            Qy = Qy.reshape(4, frame_n, int(h * mag)).transpose(1, 0, 2).copy()
            ux = ux.reshape(4, frame_n, int(w * mag)).transpose(1, 0, 2).copy()
            uy = uy.reshape(4, frame_n, int(h * mag)).transpose(1, 0, 2).copy()
        return Qx, Qy, ux, uy

    global_Qx, global_Qy, global_ux, global_uy = generate_q_u_matrix(xCoordinate, yCoordinate)
    mat_temp = cp.empty((frame_n, int(w * mag), h), dtype=cp.float32)
    threads_per_block = (1, 16, 16)
    kernel_file = get_kernel_path("catmull_rom_zoom.cu")
    config1 = {"FRAME_N": frame_n, "W": w, "H": h, "MAG": mag}
    config2 = {"FRAME_N": frame_n, "W": h, "H": int(w * mag), "MAG": mag}
    code1 = read_cu_code(kernel_file, params=config1)
    code2 = read_cu_code(kernel_file, params=config2)
    compile_options = ("--use_fast_math",)
    _cmlr_zoom_x_T = cp.RawKernel(code1, "cmlr_zoom_x_T", options=compile_options)
    _cmlr_zoom_y_T = cp.RawKernel(code2, "cmlr_zoom_x_T", options=compile_options)

    def catmullrom_zoom(image_mat: cp.array, out: cp.array, drift=None) -> cp.array:
        """the function to zoom image matrix"""

        if drift is not None:
            drift_x, drift_y = drift
            y_coordinate = cp.expand_dims(cp.linspace(0, h - 1 / mag, int(h * mag), dtype=cp.float32), 0).repeat(
                frame_n, axis=0)
            x_coordinate = cp.expand_dims(cp.linspace(0, w - 1 / mag, int(w * mag), dtype=cp.float32), 0).repeat(
                frame_n, axis=0)
            x_coordinate += cp.expand_dims(cp.asarray(drift_x), 1)
            y_coordinate += cp.expand_dims(cp.asarray(drift_y), 1)
            Qx, Qy, ux, uy = generate_q_u_matrix(x_coordinate, y_coordinate)
            blocks_per_grid = (frame_n, ceil(h / 16), ceil(int(w * mag) / 16))
            _cmlr_zoom_x_T(blocks_per_grid, threads_per_block, (image_mat, ux, Qx, Qx.shape[0], mat_temp))
            blocks_per_grid = (frame_n, ceil(int(w * mag) / 16), ceil(int(h * mag) / 16))
            _cmlr_zoom_y_T(blocks_per_grid, threads_per_block, (mat_temp, uy, Qy, Qx.shape[0], out))
        else:
            Qx, Qy, ux, uy = global_Qx, global_Qy, global_ux, global_uy
            # the name "cmlr_zoom_x_T" stands for catmullrom_zoom_in_x_direction_then_transpose_kernel
            # after transposing, the image needs to be put into this kernel function again (with different config),
            # zoomed both in height and width direction
            # do horizontal interpolation
            blocks_per_grid = (frame_n, ceil(h / 16), ceil(int(w * mag) / 16))
            _cmlr_zoom_x_T(blocks_per_grid, threads_per_block, (image_mat, ux, Qx, 1, mat_temp))
            # do vertical interpolation
            blocks_per_grid = (frame_n, ceil(int(w * mag) / 16), ceil(int(h * mag) / 16))
            _cmlr_zoom_y_T(blocks_per_grid, threads_per_block, (mat_temp, uy, Qy, 1, out))

    return catmullrom_zoom


def lanczos4_zoom_wrapper(images: cp.array, mag: float):
    """closure. It will return a function to zoom the images, but the parameters will not be calculated again"""
    # init the parameters for lanczos image resize afterwards
    h, w = images.shape[1:3]
    lanczos4_core_lut = generate_lanczos4_weights_lut()
    yCoordinate = cp.linspace(0, h - 1 / mag, int(h * mag), dtype=cp.float32)
    t, u = cp.modf(yCoordinate)
    u = u.astype(int)  # select 8 sampling points
    uy = [
        cp.maximum(u - 3, 0),
        cp.maximum(u - 2, 0),
        cp.maximum(u - 1, 0),
        cp.minimum(u, h - 1),
        cp.minimum(u + 1, h - 1),
        cp.minimum(u + 2, h - 1),
        cp.minimum(u + 3, h - 1),
        cp.minimum(u + 4, h - 1),
    ]
    Q = cp.take(lanczos4_core_lut, (t * 1024).astype(int), axis=0)
    Qy = [cp.take(Q, i, axis=1) for i in range(8)]
    xCoordinate = cp.linspace(0, w - 1 / mag, int(w * mag), dtype=cp.float32)
    del t, u, xCoordinate
    t, u = cp.modf(xCoordinate)
    u = u.astype(int)  # select 8 sampling points
    ux = [
        cp.maximum(u - 3, 0),
        cp.maximum(u - 2, 0),
        cp.maximum(u - 1, 0),
        cp.minimum(u, w - 1),
        cp.minimum(u + 1, w - 1),
        cp.minimum(u + 2, w - 1),
        cp.minimum(u + 3, w - 1),
        cp.minimum(u + 4, w - 1),
    ]
    Q = cp.take(lanczos4_core_lut, (t * 1024).astype(int), axis=0)
    Qx = [cp.take(Q, i, axis=1) for i in range(8)]
    del t, u, yCoordinate, Q, lanczos4_core_lut

    def lanczos4_zoom(image_mat: cp.array) -> cp.array:
        """the function to zoom image matrix"""
        number_of_files = image_mat.shape[0]
        # First interpolate in Y direction
        mat_temp = cp.zeros((number_of_files, w, int(h * mag)), dtype=cp.float32)
        for Qi, ui in zip(Qy, uy):
            cp.add(mat_temp, cp.transpose(cp.take(image_mat, ui, axis=1), (0, 2, 1)) * Qi, out=mat_temp)
        del image_mat
        # Then interpolate in X direction
        mat_zoomed = cp.zeros((number_of_files, int(h * mag), int(w * mag)), dtype=cp.float32)
        for Qi, ui in zip(Qx, ux):
            cp.add(mat_zoomed, cp.transpose(cp.take(mat_temp, ui, axis=1), (0, 2, 1)) * Qi, out=mat_zoomed)
        del mat_temp
        return mat_zoomed

    return lanczos4_zoom


def calculate_gradient_cupy(image_mat: cp.array, gradient_x: cp.array, gradient_y: cp.array):
    cp.subtract(image_mat[:, :, :-2], image_mat[:, :, 2:], out=gradient_x[:, :, 1:-1])
    cp.subtract(image_mat[:, :-2, :], image_mat[:, 2:, :], out=gradient_y[:, 1:-1, :])


def calculate_gradient(image_mat: cp.array, gradient_x: cp.array, gradient_y: cp.array):
    cp.subtract(image_mat[:, :, :-2], image_mat[:, :, 2:], out=gradient_x[:, :, 1:-1])
    cp.subtract(image_mat[:, :-2, :], image_mat[:, 2:, :], out=gradient_y[:, 1:-1, :])


def calculate_radiality_wrapper(images: cp.array,
                                sample_n: int,
                                mag: float,
                                radius: float,
                                do_gw: bool,
                                do_iw: bool,
                                ):
    """closure. It will return a function to calculate radiality, but the parameters will not be calculated again"""

    # generate_sample_points
    def __generate_sample_pts_rel_xy():
        if radius * mag < 1:
            warnings.warn("Product of radius and mag is less than 1. The result may be erroneous.")
        from math import sin, cos, pi
        phases = [pi / sample_n * i for i in range(sample_n)]
        ret = [(round(radius * mag * sin(phi)), round(radius * mag * cos(phi))) for phi in phases]
        ret += [(-round(radius * mag * sin(phi)), -round(radius * mag * cos(phi))) for phi in phases]
        return ret

    sample_pts_rel_xy = __generate_sample_pts_rel_xy()
    threads_per_block = (1, 16, 16)
    kernel_file = get_kernel_path("cal_radiality.cu")
    h, w = images.shape[1:3]
    h = int(h * mag)
    w = int(w * mag)
    config = {"W": w, "H": h, "SAMPLE_N": 2 * sample_n, "DO_GW": do_gw, "DO_IW": do_iw, "RADIUS": radius,
              "SAMPLE_PTS_REL_XY": "{" + "".join(["%s,%s," % each for each in sample_pts_rel_xy]) + "}"}
    code = read_cu_code(kernel_file, params=config)
    compile_options = ("--use_fast_math",)
    _cal_radiality = cp.RawKernel(code, "cal_radiality", options=compile_options)
    print("do_iw", do_iw)
    print("do_gw", do_gw)

    def calculate_radiality(image_mat: cp.array, gradient_x: cp.array, gradient_y: cp.array, result: cp.array):
        frame_n = image_mat.shape[0]
        blocks_per_grid = (frame_n, ceil(h / 16), ceil(w / 16))
        _cal_radiality(blocks_per_grid, threads_per_block,
                       (image_mat, gradient_x, gradient_y, result))

    return calculate_radiality


def calculate_trac(image_mat, out: cp.array, delay: int = 1, order: int = 0) -> cp.array:
    """Temporal radiality auto-cumulant"""

    frame_n, h, w = image_mat.shape[:3]

    _trac2 = cp.ReductionKernel(
        "T X1, T X2, T length",
        "T out",
        "X1 * X2",
        "a + b",
        "out = a / length",
        "0",
        "trac2",
    )
    _trac3 = cp.ReductionKernel(
        "T X1, T X2, T X3, T length",
        "T out",
        "X1 * X2 * X3",
        "a + b",
        "out = a / length",
        "0",
        "trac3",
    )
    _trac4 = cp.ReductionKernel(
        "T X1, T X2, T X3, T X4, T length",
        "T out",
        "X1 * X2 * X3 * X4",
        "a + b",
        "out = a / length",
        "0",
        "trac4",
    )
    _subtract_product = cp.ElementwiseKernel(
        "T A, T B",
        "T C",
        "C -= A * B",
        "subctract_product",
        no_return=True
    )
    if 2 * delay > frame_n and order != 0:
        order = 0
        warnings.warn("Total number of frames is too small to do TRAC, using TRA instead")
    deltaRt = image_mat - cp.mean(image_mat, axis=0)

    if order is 0:
        # TRA(time average)
        result = cp.mean(image_mat, axis=0)
    elif order is 2:
        # TRAC2
        if delay is 0:
            A = B = deltaRt
        else:
            A = deltaRt[:-delay]
            B = deltaRt[delay:]
        result = _trac2(A, B, frame_n, axis=0)
    elif order is 3:
        # TRAC3
        if delay is 0:
            A = B = C = deltaRt
        else:
            A = deltaRt[:-2 * delay]
            B = deltaRt[delay:-delay]
            C = deltaRt[2 * delay:]
        result = _trac3(A, B, C, frame_n, axis=0)
    elif order is 4:
        # TRAC4
        if delay is 0:
            A = B = C = D = deltaRt
        else:
            A = deltaRt[:-3 * delay]
            B = deltaRt[delay:-2 * delay]
            C = deltaRt[2 * delay:-delay]
            D = deltaRt[3 * delay:]
        result = _trac4(A, B, C, D, frame_n, axis=0)
        AB = _trac2(A, B, frame_n, axis=0)
        CD = _trac2(C, D, frame_n, axis=0)
        _subtract_product(AB * CD, result)
        del AB, CD
        AC = _trac2(A, C, frame_n, axis=0)
        BD = _trac2(B, D, frame_n, axis=0)
        _subtract_product(AC * BD, result)
        del AC, BD
        AD = _trac2(A, D, frame_n, axis=0)
        BC = _trac2(B, C, frame_n, axis=0)
        _subtract_product(AD * BC, result)
        del AD, BC
    else:
        raise ValueError("sofi-order can only be 2, 3, 4 or 0!")
    # result -= cp.min(result)
    cp.abs(result, out=out)


def to_pinned_memory(array):
    # send image matrix from CPU memory to GPU memory
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    src = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    src[...] = array
    return src


def drift_correction_wrapper(images: cp.array):
    """closure. It will return a funtion to calculate """
    base_frame = cp.array(images[0, :, :])
    base_frame_fft = cp.fft.fft2(base_frame).conj()
    shape = base_frame.shape

    def calculate_drift_offset(image_mat: cp.array):
        frame_n, h, w = image_mat.shape[:3]
        x = np.arange(w)
        y = np.arange(h)
        bounds = [(0, h), (0, w)]

        def find_max_loc_in_map(cr_map: np.array, rough_yx: cp.array):
            # 调用scipy的插值和优化寻找亚像素最大值
            precise_max_loc = np.empty((2, frame_n), dtype=cp.float32)
            for i in range(frame_n):
                np_cr_map = cr_map[i]
                F2 = interpolate.interp2d(x, y, -np_cr_map, kind="cubic")
                X0 = rough_yx[i]
                precise_max_loc[:, i] = optimize.minimize(lambda arg: F2(*arg), X0, bounds=bounds).x
            precise_max_loc[0, :] -= w // 2  # X
            precise_max_loc[1, :] -= h // 2  # Y
            return precise_max_loc

        result = cp.abs(cp.fft.fftshift(cp.fft.ifft2(cp.fft.fft2(image_mat) * base_frame_fft)))
        rough_max_loc = np.array(cp.unravel_index(cp.argmax(result.reshape(frame_n, -1), axis=1), shape)).T
        result = find_max_loc_in_map(cp.asnumpy(result), rough_max_loc)
        avg_offset = cp.average(cp.array(result), axis=0)
        print("average offset: x:", avg_offset[0], ", y:", avg_offset[1])
        return result

    return calculate_drift_offset


def srrf(images: np.array,
         mag: float = 5,
         radius: float = 0.4,
         sample_n: int = 12,
         do_gw: bool = False,
         do_iw: bool = True,
         sofi_delay: int = 1,
         sofi_order: int = 2,
         avg_every_n: int = 1,
         do_drift_correction: bool = False,
         signal=None,
         ) -> cp.array:
    # designating memory pool for async data loading (deprecated)
    # ban memory pool for memory safety
    cp.cuda.set_allocator(None)
    cp.cuda.set_pinned_memory_allocator(None)
    # clean memory pool
    # cp.get_default_memory_pool().free_all_blocks()
    # pinned_memory_pool.free_all_blocks()
    # preparing
    frame_n, h, w = shape = images.shape[:3]
    batch_size = get_batch_size(frame_n, shape, mag)
    if batch_size is 0:
        raise ValueError("Image shape is too large to process!")
    if batch_size * avg_every_n > frame_n:
        if avg_every_n < frame_n:
            batch_size = frame_n // avg_every_n * avg_every_n
        else:
            avg_every_n = frame_n
            batch_size = frame_n
            warnings.warn("avg_every_n is greater than total number of frames. All frames will be averaged.")
    else:
        batch_size = batch_size * avg_every_n
    result = cp.zeros((int(h * mag), int(w * mag)), dtype=cp.float64)
    gradientX = cp.zeros((ceil(batch_size / avg_every_n), h, w), dtype=cp.float32)
    gradientY = cp.zeros((ceil(batch_size / avg_every_n), h, w), dtype=cp.float32)
    gradientX_zoomed = cp.zeros((ceil(batch_size / avg_every_n), int(h * mag), int(w * mag)), dtype=cp.float32)
    gradientY_zoomed = cp.zeros((ceil(batch_size / avg_every_n), int(h * mag), int(w * mag)), dtype=cp.float32)
    image_mat_zoomed = cp.empty((ceil(batch_size / avg_every_n), int(h * mag), int(w * mag)), dtype=cp.float32)
    radiality = cp.empty((ceil(batch_size / avg_every_n), int(h * mag), int(w * mag)), dtype=cp.float32)
    trac_mat = cp.empty((int(h * mag), int(w * mag)), dtype=cp.float64)

    drift_correction = drift_correction_wrapper(images) if do_drift_correction else None
    catmullrom_zoom = catmullrom_zoom_wrapper(images, mag, ceil(batch_size / avg_every_n))
    calculate_radiality = calculate_radiality_wrapper(images, sample_n, mag, radius, do_gw, do_iw)

    # init CUDA streams
    sm_n = 2  # number of stream managers used in graphic card.
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(sm_n)]
    mats = [cp.empty(images[:batch_size].shape, images[:batch_size].dtype) for _ in range(sm_n)]

    for i, image_mat in enumerate(split_image_sequence(images, batch_size)):
        # send signal for gui progress bar
        if signal is not None:
            signal.emit(i * batch_size * 100 / frame_n)

        # dispatch the task of copying image mat from CPU to GPU asynchronously
        # select 2 streams

        index_next = i % sm_n
        index_current = (i - 1) % sm_n
        # Use next stream to load image data in advance
        if image_mat is not None:
            np_mat = to_pinned_memory(image_mat)
            mats[index_next].set(np_mat, stream=streams[index_next])

        # skip the first time(the copy is not finished yet)
        if i is 0:
            continue
        # Use the current stream to process the image mat on GPU
        image_mat = mats[index_current]
        with streams[index_current]:
            print("dispatching %d/%d" % (i * batch_size, frame_n))
            # average every n images to reduce the calculation. Optional
            if avg_every_n != 1:
                image_mat = avg_every_n_frames(image_mat, avg_every_n)
            calculate_gradient(image_mat, gradientX, gradientY)
            # 这里梯度也必须是原始图像的梯度插值得到。如果直接使用放大后的图像计算梯度，会受到插值算法的影响出现方格状图案
            drift = drift_correction(image_mat) if do_drift_correction and image_mat.shape[0] > 1 else None
            catmullrom_zoom(image_mat, image_mat_zoomed, drift)
            catmullrom_zoom(gradientX, gradientX_zoomed, drift)
            catmullrom_zoom(gradientY, gradientY_zoomed, drift)
            # calculate srrf radiality in every frame
            calculate_radiality(image_mat_zoomed, gradientX_zoomed, gradientY_zoomed, radiality)
            # do TRAC calculation (sofi)
            calculate_trac(radiality, trac_mat, sofi_delay, sofi_order)
            cp.add(result, trac_mat, out=result)
            streams[index_current].synchronize()
        # The next stream will be started only if the current stream is done
        # This is to protect the data on the next stream, otherwise the data will be
        # overwritten and the result will be wrong
        stop_event = streams[index_current].record()
        streams[index_next].wait_event(stop_event)

    [stream.synchronize() for stream in streams]

    return cp.asnumpy(result)


def get_batch_size(n: int,
                   shape: tuple,
                   m: float,
                   ) -> int:
    """find the max batch_size for image sequence"""
    batch_size: int = n
    h: int = shape[1] * m
    w: int = shape[2] * m

    n = 32  # 表示能开n个放大后图像那么大的矩阵，如果数量小了虽然不会立刻崩溃，但会增加内存分配和回收的次数影响性能
    available_memory = cp.cuda.Device(0).mem_info[0]
    batch_size = min(batch_size, int(available_memory / (h * w * 2 * n)))
    print("batch_size", batch_size)
    return batch_size


def read_images_from_folder(folder_path: str,
                            signal=None
                            ) -> np.array:
    """read a part of all images from a folder to a generator"""

    if folder_path[-1] is "/" or folder_path[-1] is "\\":
        folder_path = folder_path[:-1]
    files = os.listdir(folder_path)
    image_extension_names = [".jpg", ".jpeg",
                             ".gif", ".bmp", ".png", ".tif", ".tiff", ".webp"]
    files = list(
        filter(lambda file: not os.path.isdir(file) and os.path.splitext(file)[-1] in image_extension_names, files))
    if signal is None:
        images = np.array([np.array(Image.open(folder_path + "/" + img).convert("L")) for img in files], dtype=np.float32)
    else:
        images = []
        total_number = len(files)
        for i, img in enumerate(files):
            images.append(np.array(Image.open(folder_path + "/" + img).convert("L"), dtype=np.float32))
            signal.emit(i * 100 // total_number)
        images = np.array(images)
    if images.ndim is 1:
        warnings.warn("Invalid image is contained in the folder! Please check the image folder, and make sure all"
                      " images are frames of image sequence.")
        return None

    return images


def read_images_from_tiff_file(file_path: str,
                               signal=None,
                               ) -> np.array:
    if signal is not None:
        signal.emit(10)
    img = Image.open(file_path)
    n_frames = img.n_frames
    images = np.empty((n_frames, img.height, img.width), dtype=np.float32)
    for i, frame in enumerate(ImageSequence.Iterator(img)):
        images[i] = frame
        if signal is not None:
            signal.emit(10 + i * 90 // n_frames)
    return images


def arg_parser() -> tuple:
    """deal with cli commands. Load Image and call srrf function. Return original image mat and srrf-ed mat"""
    parser = argparse.ArgumentParser(description="An implementation of srrf super-resolution algorithm")
    parser.add_argument("data_path", type=str, help="path to image sequence folder or a multi-frame tif image")
    parser.add_argument("-m", dest="mag", type=float, default=4, help="set magnification rate")
    parser.add_argument("-r", dest="radius", type=float, default=0.4, help="set radius of sampling circle")
    parser.add_argument("-sn", dest="sample_n", type=int, default=6, help="number of sampling points")
    parser.add_argument("--dogw", dest="do_gw", action="store_const", const=True, default=False,
                        help="multiply gradient magnitude with result")
    parser.add_argument("--noiw", dest="do_iw", action="store_const", const=False, default=True,
                        help="DON\"T multiply the image magnitude with result")
    parser.add_argument("-sd", dest="sofi_delay", type=int, default=1, help="delay frames of sofi")
    parser.add_argument("-so", dest="sofi_order", type=int, default=2, help="sofi order (2~4)")
    parser.add_argument("--avg", dest="avg_every_n", type=int, default=1,
                        help="average every n frames into a single one")
    parser.add_argument("--dodc", dest="do_drift_correction", action="store_const", const=True, default=False,
                        help="do drift correction")
    args = vars(parser.parse_args())
    filename = args["data_path"]
    img_array = None
    if os.path.isdir(filename):
        img_array = read_images_from_folder(filename)
    elif os.path.isfile(filename) and os.path.splitext(filename)[-1] in (".tif", ".tiff"):
        img_array = read_images_from_tiff_file(filename)
    else:
        raise ValueError("Image sequence folder or TIF file not found!")
    del args["data_path"]
    time1 = time.time()
    ret_img_array = srrf(img_array, **args)
    time2 = time.time()
    print("time cost:", time2 - time1)
    return img_array, ret_img_array


def plot_result(img_array, ret_img_array):
    std = float(cp.std(ret_img_array))
    gamma = [0.5]
    v_max = [8 * std]
    frac = ret_img_array.shape[0] / 128
    thumbnail_shape = [int(each / frac) for each in ret_img_array.shape]
    thumbnail = np.array(Image.fromarray(ret_img_array).resize(thumbnail_shape, Image.ANTIALIAS))
    # first select proper vmax and gamma
    fig = plt.figure()
    ima = plt.axes((0.1, 0.20, 0.8, 0.7))
    ima.set_title("Please select proper Vmax and Gamma for result")
    ima.imshow(thumbnail, cmap="gray", norm=colors.PowerNorm(gamma=gamma[0]), vmin=0, vmax=v_max[0])
    # plot slider
    om1 = plt.axes((0.25, 0.05, 0.65, 0.03))
    om2 = plt.axes((0.25, 0.10, 0.65, 0.03))
    som1 = Slider(om1, r"$max value$", 0, 16 * std, valinit=v_max[0], dragging=True)  # generate the first slider
    som2 = Slider(om2, r"$Gamma$", 0, 4.0, valinit=gamma[0], dragging=True)  # generate the second slider

    # define update function for slides
    def update(val):
        s1 = som1.val
        s2 = som2.val
        v_max[0] = s1
        gamma[0] = s2
        ima.imshow(thumbnail, cmap="gray", norm=colors.PowerNorm(gamma=gamma[0]), vmin=0, vmax=v_max[0])
        fig.canvas.draw_idle()

    # bind update function to slides
    som1.on_changed(update)
    som2.on_changed(update)
    plt.show()
    del thumbnail

    # then print the final compared images
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # plot original image sequence
    ax1.set_title("Original image sequence")
    ims = []
    for frame in img_array[:200]:
        im = ax1.imshow(frame, cmap="gray", animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=0)
    # animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=0)
    # plot result
    ax2.imshow(ret_img_array, cmap="gray", norm=colors.PowerNorm(gamma=gamma[0]), vmin=0, vmax=v_max[0])
    ax2.set_title("srrf image")

    plt.show()


"""
def plot_result_3d(data_mat):
    # draw 3d figure
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    # ax.view_init(azim=0) # 用于控制显示角度
    xs, ys = np.meshgrid(np.arange(data_mat.shape[1]), np.arange(data_mat.shape[0]))
    # 生成坐标轴，注意两个坐标顺序要反下
    # 反的原因是在numpy的表示体系里，第一个维度是y轴，第二个维度才是x轴
    ax.plot_surface(xs, ys, data_mat, cmap="gist_earth", rcount=64, ccount=64)
    plt.xlabel("x"), plt.ylabel("y")
    plt.show()
"""


def main():
    # 性能分析:nvprof -f -o srrf.nvvp python srrf.py ...
    # 之后打开srrf.nvvp

    # Usage example:
    # python srrf.py -h                 Print help
    # python srrf.py example.tif        Reconstruct selected tif image
    # python srrf.py sequence_folder    Reconstruct all image sequence in a folder
    # WARNING: file name should not contain space
    # you can also set sofi_order, ring_radius, etc. See help for details.

    imgArray, retImgArray = arg_parser()
    plot_result(imgArray, retImgArray)


if __name__ == "__main__":
    main()
