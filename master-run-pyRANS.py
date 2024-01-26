# Original CFD-solver written by Lars Davidsson, https://www.tfd.chalmers.se/~lada/pyCALC-RANS.html
# This derivative of Lars Davidsson work is distributed with permission
# Written for TRACKS course TRA220, https://www.tfd.chalmers.se/~lada/CUDA-tracks.html
# This is a accelerated version written by Viktor Sundström and Joar Forsberg, dated 2023-01-20
# Run the batch files using $ python exec-pyCALC-RANS.py NAME_OF_BATCHFILE (written without .py extension) 
# Using supplied .py configfiles makes it easier to deploy and run large batches on compute cluster

print("Running ... ")

from scipy import sparse
import numpy
import sys
import time
import pyamg
import socket
import scipy
import json
import os
import resource
import scipy.io as sio
import sys
from IPython import display
import torch
import sys
import pynvml

# Resolutions, created with supplied file
# x12909y3872
# x5773y1732
# x4082y1224
# x1825y547
# x577y173
# x300y90
# x182y54
# x57y17

# Solvers, not all supported in Cupy
# bicg (BIConjugate Gradient)
# bicgstab (BIConjugate Gradient STABilized)
# cg (Conjugate Gradient) - symmetric positive definite matrices only
# cgs (Conjugate Gradient Squared)
# gmres (Generalized Minimal RESidual)
# minres (MINimum RESidual)
# qmr (Quasi-Minimal Residual Method)
# jacobi_dense (naïve implementation of algortihm, not accounting for sparse data type
# jacobi (Implementation based on the sparse data type)

# Precision
# fp16 (has issues)
# fp32
# fp64
# fp128 (has issues)

print(sys.argv[1:])

batchArr = []
conf_file = str(sys.argv[1])
if conf_file:
    data_imported = __import__(conf_file)
    batchArr = data_imported.batchArr
else:
    print("No run file supplied")

dry_run = False

print(f"Dry run {dry_run} on {conf_file}\n{batchArr}")


# Loop over all configurations
print(f"Running {len(batchArr)} simulations")
for conf in batchArr:
    print(f"Current settings: {conf}")
    maxit=1
    if not dry_run:
        maxit = conf['maxIter'] 
    mesh = conf['mesh']
    # True if every numpy should be replaced with cupy version
    useCupy = conf['useCupy']

    # True if pyamgx is supported on system, makes option available
    usePyamgx = conf['usePyamgx']


    solver_vel = conf['solver_vel'] #"lgmres"
    solver_pp = conf['solver_pp']  #"pyamg"
    solver_turb = conf['solver_turb'] #"lgmres" # cgs, gmres, cg should work, jacobi(x)

    if useCupy:
        import cupy as np
        import cupyx
        from cupyx.scipy.sparse import spdiags, eye, linalg
        # Need equivlent of scipy spdiags linalg eye
        pynvml.nvmlInit()
        # Profiling VRAM

    else:
        import numpy as np
        from scipy.sparse import spdiags, linalg, eye
    if usePyamgx:
        import pyamgx
        pyamgx.initialize()
        if not useCupy:
            pynvml.nvmlInit()

    if conf['precision'] == 'fp16':
        precision = np.float16
        print("Not completely supported by cupy-scipy, will yield convergence problems")
    elif conf['precision'] == 'fp32':
        precision = np.float32
    elif conf['precision'] == 'fp64':
        precision = np.float64
    elif conf['precision'] == 'fp128':
        precision = np.float128
        print("Not completely supported by cupy-scipy")
        
    # Grouping all CUDA-numerically uninteresting function in one indent
    if True:
        def setup_case():
            global c_omega_1, c_omega_2, cmu, convergence_limit_eps, convergence_limit_k, convergence_limit_om, convergence_limit_pp, convergence_limit_u, convergence_limit_v, convergence_limit_w, dist, fx, fy, imon, jmon, kappa, k_bc_east, k_bc_east_type, k_bc_north, k_bc_north_type, k_bc_south, k_bc_south_type, k_bc_west, k_bc_west_type, kom, maxit, ni, nj, nsweep_kom, nsweep_pp, nsweep_vel, om_bc_east, om_bc_east_type, om_bc_north, om_bc_north_type, om_bc_south, om_bc_south_type, om_bc_west, om_bc_west_type, p_bc_east, p_bc_east_type, p_bc_north, p_bc_north_type, p_bc_south, p_bc_south_type, p_bc_west, p_bc_west_type, prand_k, prand_omega, resnorm_p, resnorm_vel, restart, save, save_vtk_movie, scheme, scheme_turb, solver_pp, solver_vel, solver_turb, sormax, u_bc_east, u_bc_east_type, u_bc_north, u_bc_north_type, u_bc_south, u_bc_south_type, u_bc_west, u_bc_west_type, urfvis, urf_vel, urf_k, urf_p, urf_omega, v_bc_east, v_bc_east_type, v_bc_north, v_bc_north_type, v_bc_south, v_bc_south_type, v_bc_west, v_bc_west_type, viscos, vol, vtk, vtk_save, vtk_file_name, x2d, xp2d, y2d, yp2d

            ########### section 1 choice of differencing scheme ###########
            scheme = "h"  # hybrid
            scheme_turb = "h"  # hybrid upwind-central

            ########### section 2 turbulence models ###########
            cmu = 0.09
            kom = True
            c_omega_1 = 5.0 / 9.0
            c_omega_2 = 3.0 / 40.0
            prand_omega = 2.0
            prand_k = 2.0

            ########### section 3 restart/save ###########
            restart = False
            save = False

            ########### section 4 fluid properties ###########
            viscos = 3.57e-5

            ########### section 5 relaxation factors ###########
            urfvis = 0.5
            urf_vel = 0.5
            urf_k = 0.5
            urf_p = 0.5
            urf_omega = 0.5

            ########### section 6 number of iteration and convergence criterira ###########
            min_iter = 1
            sormax = 1e-5

            nsweep_vel = 50
            nsweep_pp = 50
            nsweep_kom = 50
            convergence_limit_u = 1e-6
            convergence_limit_v = 1e-6
            convergence_limit_k = 1e-6
            convergence_limit_om = 1e-8
            convergence_limit_pp = 1e-6

            ########### section 7 all variables are printed during the iteration at node ###########
            imon = ni - 10
            jmon = int(nj / 2)

            ########### section 8 save data for post-processing ###########
            vtk = False
            save_all_files = False
            vtk_file_name = "bound"

            ########### section 9 residual scaling parameters ###########
            uin = 1
            resnorm_p = uin * y2d[1, -1]
            resnorm_vel = uin**2 * y2d[1, -1]

            ########### Section 10 boundary conditions ###########

            # boundary conditions for u
            u_bc_west = np.ones(nj,dtype=precision)
            u_bc_east = np.zeros(nj,dtype=precision)
            u_bc_south = np.zeros(ni,dtype=precision)
            u_bc_north = np.zeros(ni,dtype=precision)

            u_bc_west_type = "d"
            u_bc_east_type = "n"
            u_bc_south_type = "d"
            u_bc_north_type = "n"

            # boundary conditions for v
            v_bc_west = np.zeros(nj,dtype=precision)
            v_bc_east = np.zeros(nj,dtype=precision)
            v_bc_south = np.zeros(ni,dtype=precision)
            v_bc_north = np.zeros(ni,dtype=precision)

            v_bc_west_type = "d"
            v_bc_east_type = "n"
            v_bc_south_type = "d"
            v_bc_north_type = "d"

            # boundary conditions for p
            p_bc_west = np.zeros(nj,dtype=precision)
            p_bc_east = np.zeros(nj,dtype=precision)
            p_bc_south = np.zeros(ni,dtype=precision)
            p_bc_north = np.zeros(ni,dtype=precision)

            p_bc_west_type = "n"
            p_bc_east_type = "n"
            p_bc_south_type = "n"
            p_bc_north_type = "n"

            # boundary conditions for k
            k_bc_west = np.ones(nj,dtype=precision) * 1e-2
            k_bc_west[10:] = 1e-5
            k_bc_east = np.zeros(nj,dtype=precision)
            k_bc_south = np.zeros(ni,dtype=precision)
            k_bc_north = np.ones(ni,dtype=precision) * 1e-5

            k_bc_west_type = "d"
            k_bc_east_type = "n"
            k_bc_south_type = "d"
            k_bc_north_type = "n"

            # boundary conditions for omega
            om_bc_west = np.ones(nj,dtype=precision)
            om_bc_east = np.zeros(nj,dtype=precision)
            om_bc_south = np.zeros(ni,dtype=precision)
            om_bc_north = np.zeros(ni,dtype=precision)

            xwall_s = 0.5 * (x2d[0:-1, 0] + x2d[1:, 0])
            ywall_s = 0.5 * (y2d[0:-1, 0] + y2d[1:, 0])
            dist2_s = (yp2d[:, 0] - ywall_s) ** 2 + (xp2d[:, 0] - xwall_s) ** 2
            om_bc_south = 6 * viscos / 0.075 / dist2_s

            om_bc_west_type = "d"
            om_bc_east_type = "n"
            om_bc_south_type = "d"
            om_bc_north_type = "n"

            return
        def modify_init(u2d, v2d, k2d, om2d, vis2d):
            # set inlet field in entre domain
            u2d = np.repeat(u_bc_west[None, :], repeats=ni, axis=0)
            k2d = np.repeat(k_bc_west[None, :], repeats=ni, axis=0)
            om2d = np.repeat(om_bc_west[None, :], repeats=ni, axis=0)
            vis2d = k2d / om2d + viscos

            return u2d, v2d, k2d, om2d, vis2d
        def modify_inlet():
            global y_rans, y_rans, u_rans, v_rans, k_rans, om_rans, uv_rans, k_bc_west, eps_bc_west, om_bc_west

            return u_bc_west, v_bc_west, k_bc_west, om_bc_west, u2d_face_w, convw
        def modify_conv(convw, convs):
            convs[:, 0, :] = 0
            convs[:, -1, :] = 0

            return convw, convs
        def modify_u(su2d, sp2d):
            global file1

            su2d[0, :] = su2d[0, :] + convw[0, :] * u_bc_west
            sp2d[0, :] = sp2d[0, :] - convw[0, :]
            vist = (
                vis2d[
                    0,
                    :,
                ]
                - viscos
            )
            su2d[0, :] = su2d[0, :] + vist * aw_bound * u_bc_west
            sp2d[0, :] = sp2d[0, :] - vist * aw_bound
            # Dump information in terminal and save historical data, manually commended out
            return su2d, sp2d
        def modify_v(su2d, sp2d):
            su2d[0, :] = su2d[0, :] + convw[0, :] * v_bc_west
            sp2d[0, :] = sp2d[0, :] - convw[0, :]
            vist = vis2d[0, :] - viscos
            su2d[0, :] = su2d[0, :] + vist * aw_bound * v_bc_west
            sp2d[0, :] = sp2d[0, :] - vist * aw_bound

            return su2d, sp2d
        def modify_k(su2d, sp2d):
            su2d[0, :] = su2d[0, :] + np.maximum(convw[0, :], 0) * k_bc_west
            sp2d[0, :] = sp2d[0, :] - convw[0, :]
            vist = vis2d[0, :] - viscos
            su2d[0, :] = su2d[0, :] + vist * aw_bound * k_bc_west
            sp2d[0, :] = sp2d[0, :] - vist * aw_bound

            return su2d, sp2d
        def modify_om(su2d, sp2d):
            su2d[0, :] = su2d[0, :] + np.maximum(convw[0, :], 0) * om_bc_west
            sp2d[0, :] = sp2d[0, :] - convw[0, :]
            vist = vis2d[0, :] - viscos
            su2d[0, :] = su2d[0, :] + vist * aw_bound * om_bc_west
            sp2d[0, :] = sp2d[0, :] - vist * aw_bound

            return su2d, sp2d
        def modify_outlet(convw):
            # inlet
            flow_in = np.sum(convw[0, :])
            flow_out = np.sum(convw[-1, :])
            area_out = np.sum(areaw[-1, :])

            uinc = (flow_in - flow_out) / area_out
            ares = areaw[-1, :]
            convw[-1, :] = convw[-1, :] + uinc * ares

            flow_out_new = np.sum(convw[-1, :])

            print(
                f"{'flow_in: '} {flow_in:.3e},{'  flow_out: '} {flow_out:.3e},{'  flow_out_new: '} {flow_out_new:.3e},{'  uinc: '} {uinc:.3e}"
            )

            return convw
        def fix_omega():
            aw2d[:, 0] = 0
            ae2d[:, 0] = 0
            as2d[:, 0] = 0
            an2d[:, 0] = 0
            ap2d[:, 0] = 1
            su2d[:, 0] = om_bc_south

            return aw2d, ae2d, as2d, an2d, ap2d, su2d, sp2d
        def modify_vis(vis2d):
            return vis2d
        def fix_k():
            return aw2d, ae2d, as2d, an2d, ap2d, su2d, sp2d
        def init():
            # distance to nearest wall
            ywall_s = 0.5 * (y2d[0:-1, 0] + y2d[1:, 0])
            dist_s = yp2d - ywall_s[:, None]
            ywall_n = 0.5 * (y2d[0:-1, -1] + y2d[1:, -1])
            dist_n = ywall_n[:, None] - yp2d
            dist = np.minimum(dist_s, dist_n)

            #  west face coordinate
            xw = 0.5 * (x2d[0:-1, 0:-1] + x2d[0:-1, 1:])
            yw = 0.5 * (y2d[0:-1, 0:-1] + y2d[0:-1, 1:])

            del1x = ((xw - xp2d) ** 2 + (yw - yp2d) ** 2) ** 0.5
            del2x = (
                (xw - np.roll(xp2d, 1, axis=0)) ** 2 + (yw - np.roll(yp2d, 1, axis=0)) ** 2
            ) ** 0.5
            fx = del2x / (del1x + del2x)

            #  south face coordinate
            xs = 0.5 * (x2d[0:-1, 0:-1] + x2d[1:, 0:-1])
            ys = 0.5 * (y2d[0:-1, 0:-1] + y2d[1:, 0:-1])

            del1y = ((xs - xp2d) ** 2 + (ys - yp2d) ** 2) ** 0.5
            del2y = (
                (xs - np.roll(xp2d, 1, axis=1)) ** 2 + (ys - np.roll(yp2d, 1, axis=1)) ** 2
            ) ** 0.5
            fy = del2y / (del1y + del2y)

            areawy = np.diff(x2d, axis=1)
            areawx = -np.diff(y2d, axis=1)

            areasy = -np.diff(x2d, axis=0)
            areasx = np.diff(y2d, axis=0)

            areaw = (areawx**2 + areawy**2) ** 0.5
            areas = (areasx**2 + areasy**2) ** 0.5

            # volume approaximated as the vector product of two triangles for cells
            ax = np.diff(x2d, axis=1)
            ay = np.diff(y2d, axis=1)
            bx = np.diff(x2d, axis=0)
            by = np.diff(y2d, axis=0)

            areaz_1 = 0.5 * np.absolute(ax[0:-1, :] * by[:, 0:-1] - ay[0:-1, :] * bx[:, 0:-1])

            ax = np.diff(x2d, axis=1)
            ay = np.diff(y2d, axis=1)
            areaz_2 = 0.5 * np.absolute(ax[1:, :] * by[:, 0:-1] - ay[1:, :] * bx[:, 0:-1])

            vol = areaz_1 + areaz_2

            # coeff at south wall (without viscosity)
            as_bound = areas[:, 0] ** 2 / (0.5 * vol[:, 0])

            # coeff at north wall (without viscosity)
            an_bound = areas[:, -1] ** 2 / (0.5 * vol[:, -1])

            # coeff at west wall (without viscosity)
            aw_bound = areaw[0, :] ** 2 / (0.5 * vol[0, :])

            ae_bound = areaw[-1, :] ** 2 / (0.5 * vol[-1, :])

            return (
                areaw,
                areawx,
                areawy,
                areas,
                areasx,
                areasy,
                vol,
                fx,
                fy,
                aw_bound,
                ae_bound,
                as_bound,
                an_bound,
                dist,
            )
        def print_indata():
            print("////////////////// Start of input data ////////////////// \n\n\n")

            print("\n\n########### section 1 choice of differencing scheme ###########")
            print(f"{'scheme: ':<29}   {scheme}")
            print(f"{'scheme_turb: ':<29}   {scheme_turb}")

            print("\n\n########### section 2 turbulence models ###########")

            print(f"{'cmu: ':<29} {cmu}")
            print(f"{'kom: ':<29} {kom}")
            if kom:
                print(f"{'c_omega_1: ':<29} {c_omega_1:.3f}")
                print(f"{'c_omega_2: ':<29} {c_omega_2}")
                print(f"{'prand_k: ':<29} {prand_k}")
                print(f"{'prand_omega: ':<29} {prand_omega}")

            print("\n\n########### section 3 restart/save ###########")
            print(f"{'restart: ':<29} {restart}")
            print(f"{'save: ':<29} {save}")

            print("\n\n########### section 4 fluid properties ###########")
            print(f"{'viscos: ':<29} {viscos:.2e}")

            print("\n\n########### section 5 relaxation factors ###########")
            print(f"{'urfvis: ':<29} {urfvis}")

            print(
                "\n\n########### section 6 number of iteration and convergence criterira ###########"
            )
            print(f"{'sormax: ':<29} {sormax}")
            print(f"{'maxit: ':<29} {maxit}")
            print(f"{'solver_vel: ':<29} {solver_vel}")
            print(f"{'solver_turb: ':<29} {solver_turb}")
            print(f"{'nsweep_vel: ':<29} {nsweep_vel}")
            print(f"{'nsweep_pp: ':<29} {nsweep_pp}")
            print(f"{'nsweep_kom: ':<29} {nsweep_kom}")
            print(f"{'convergence_limit_u: ':<29} {convergence_limit_u}")
            print(f"{'convergence_limit_v: ':<29} {convergence_limit_v}")
            print(f"{'convergence_limit_pp: ':<29} {convergence_limit_pp}")
            print(f"{'convergence_limit_k: ':<29} {convergence_limit_k}")
            print(f"{'convergence_limit_om: ':<29} {convergence_limit_om}")

            print(
                "\n\n########### section 7 all variables are printed during the iteration at node ###########"
            )
            print(f"{'imon: ':<29} {imon}")
            print(f"{'jmon: ':<29} {jmon}")

            print("\n\n########### section 8 time-averaging ###########")

            print("\n\n########### section 9 residual scaling parameters ###########")
            print(f"{'resnorm_p: ':<29} {resnorm_p:.1f}")
            print(f"{'resnorm_vel: ':<29} {resnorm_vel:.1f}")

            print("\n\n########### Section 10 grid and boundary conditions ###########")
            print(f"{'ni: ':<29} {ni}")
            print(f"{'nj: ':<29} {nj}")
            print("\n")
            print("\n")

            print("------boundary conditions for u")
            print(f"{' ':<5}{'u_bc_west_type: ':<29} {u_bc_west_type}")
            print(f"{' ':<5}{'u_bc_east_type: ':<29} {u_bc_east_type}")
            if u_bc_west_type == "d":
                print(f"{' ':<5}{'u_bc_west[0]: ':<29} {u_bc_west[0]}")
            if u_bc_east_type == "d":
                print(f"{' ':<5}{'u_bc_east[0]: ':<29} {u_bc_east[0]}")

            print(f"{' ':<5}{'u_bc_south_type: ':<29} {u_bc_south_type}")
            print(f"{' ':<5}{'u_bc_north_type: ':<29} {u_bc_north_type}")

            if u_bc_south_type == "d":
                print(f"{' ':<5}{'u_bc_south[0]: ':<29} {u_bc_south[0]}")
            if u_bc_north_type == "d":
                print(f"{' ':<5}{'u_bc_north[0]: ':<29} {u_bc_north[0]}")

            print("------boundary conditions for v")
            print(f"{' ':<5}{'v_bc_west_type: ':<29} {v_bc_west_type}")
            print(f"{' ':<5}{'v_bc_east_type: ':<29} {v_bc_east_type}")
            if v_bc_west_type == "d":
                print(f"{' ':<5}{'v_bc_west[0]: ':<29} {v_bc_west[0]}")
            if v_bc_east_type == "d":
                print(f"{' ':<5}{'v_bc_east[0]: ':<29} {v_bc_east[0]}")

            print(f"{' ':<5}{'v_bc_south_type: ':<29} {v_bc_south_type}")
            print(f"{' ':<5}{'v_bc_north_type: ':<29} {v_bc_north_type}")

            if v_bc_south_type == "d":
                print(f"{' ':<5}{'v_bc_south[0]: ':<29} {v_bc_south[0]}")
            if v_bc_north_type == "d":
                print(f"{' ':<5}{'v_bc_north[0]: ':<29} {v_bc_north[0]}")

            print("------boundary conditions for k")
            print(f"{' ':<5}{'k_bc_west_type: ':<29} {k_bc_west_type}")
            print(f"{' ':<5}{'k_bc_east_type: ':<29} {k_bc_east_type}")
            if k_bc_west_type == "d":
                print(f"{' ':<5}{'k_bc_west[0]: ':<29} {k_bc_west[0]}")
            if k_bc_east_type == "d":
                print(f"{' ':<5}{'k_bc_east[0]: ':<29} {k_bc_east[0]}")

            print(f"{' ':<5}{'k_bc_south_type: ':<29} {k_bc_south_type}")
            print(f"{' ':<5}{'k_bc_north_type: ':<29} {k_bc_north_type}")

            if k_bc_south_type == "d":
                print(f"{' ':<5}{'k_bc_south[0]: ':<29} {k_bc_south[0]}")
            if k_bc_north_type == "d":
                print(f"{' ':<5}{'k_bc_north[0]: ':<29} {k_bc_north[0]}")

            print("------boundary conditions for omega")
            print(f"{' ':<5}{'om_bc_west_type: ':<29} {om_bc_west_type}")
            print(f"{' ':<5}{'om_bc_east_type: ':<29} {om_bc_east_type}")
            if om_bc_west_type == "d":
                print(f"{' ':<5}{'om_bc_west[0]: ':<29} {om_bc_west[0]:.1f}")
            if om_bc_east_type == "d":
                print(f"{' ':<5}{'om_bc_east[0]: ':<29} {om_bc_east[0]:.1f}")

            print(f"{' ':<5}{'om_bc_south_type: ':<29} {om_bc_south_type}")
            print(f"{' ':<5}{'om_bc_north_type: ':<29} {om_bc_north_type}")

            if om_bc_south_type == "d":
                print(f"{' ':<5}{'om_bc_south[0]: ':<29} {om_bc_south[0]:.1f}")
            if om_bc_north_type == "d":
                print(f"{' ':<5}{'om_bc_north[0]: ':<29} {om_bc_north[0]:.1f}")

            print("\n\n\n ////////////////// End of input data //////////////////\n\n\n")

            return
    if True:
        def compute_face_phi(
            phi2d,
            phi_bc_west,
            phi_bc_east,
            phi_bc_south,
            phi_bc_north,
            phi_bc_west_type,
            phi_bc_east_type,
            phi_bc_south_type,
            phi_bc_north_type,
        ):
            phi2d_face_w = np.empty((ni + 1, nj),dtype=precision)
            phi2d_face_s = np.empty((ni, nj + 1),dtype=precision)
            phi2d_face_w[0:-1, :] = fx * phi2d + (1 - fx) * np.roll(phi2d, 1, axis=0)
            phi2d_face_s[:, 0:-1] = fy * phi2d + (1 - fy) * np.roll(phi2d, 1, axis=1)

            # west boundary
            phi2d_face_w[0, :] = phi_bc_west
            if phi_bc_west_type == "n":
                # neumann
                phi2d_face_w[0, :] = phi2d[0, :]

            # east boundary
            phi2d_face_w[-1, :] = phi_bc_east
            if phi_bc_east_type == "n":
                # neumann
                phi2d_face_w[-1, :] = phi2d[-1, :]
                phi2d_face_w[-1, :] = phi2d_face_w[-2, :]

            # south boundary
            phi2d_face_s[:, 0] = phi_bc_south
            if phi_bc_south_type == "n":
                # neumann
                phi2d_face_s[:, 0] = phi2d[:, 0]

            # north boundary
            phi2d_face_s[:, -1] = phi_bc_north
            if phi_bc_north_type == "n":
                # neumann
                phi2d_face_s[:, -1] = phi2d[:, -1]

            return phi2d_face_w, phi2d_face_s


        def dphidx(phi_face_w, phi_face_s):
            phi_w = phi_face_w[0:-1, :] * areawx[0:-1, :]
            phi_e = -phi_face_w[1:, :] * areawx[1:, :]
            phi_s = phi_face_s[:, 0:-1] * areasx[:, 0:-1]
            phi_n = -phi_face_s[:, 1:] * areasx[:, 1:]
            return (phi_w + phi_e + phi_s + phi_n) / vol


        def dphidy(phi_face_w, phi_face_s):
            phi_w = phi_face_w[0:-1, :] * areawy[0:-1, :]
            phi_e = -phi_face_w[1:, :] * areawy[1:, :]
            phi_s = phi_face_s[:, 0:-1] * areasy[:, 0:-1]
            phi_n = -phi_face_s[:, 1:] * areasy[:, 1:]
            return (phi_w + phi_e + phi_s + phi_n) / vol


        def coeff(convw, convs, vis2d, prand, scheme_local):
            visw = np.zeros((ni + 1, nj),dtype=precision)
            viss = np.zeros((ni, nj + 1),dtype=precision)
            vis_turb = (vis2d - viscos) / prand

            visw[0:-1, :] = fx * vis_turb + (1 - fx) * np.roll(vis_turb, 1, axis=0) + viscos
            viss[:, 0:-1] = fy * vis_turb + (1 - fy) * np.roll(vis_turb, 1, axis=1) + viscos

            volw = np.ones((ni + 1, nj),dtype=precision) * 1e-10
            vols = np.ones((ni, nj + 1),dtype=precision) * 1e-10
            volw[1:, :] = 0.5 * np.roll(vol, -1, axis=0) + 0.5 * vol
            diffw = visw[0:-1, :] * areaw[0:-1, :] ** 2 / volw[0:-1, :]
            vols[:, 1:] = 0.5 * np.roll(vol, -1, axis=1) + 0.5 * vol
            diffs = viss[:, 0:-1] * areas[:, 0:-1] ** 2 / vols[:, 0:-1]

            if scheme_local == "h":
                if iter == 0:
                    print("hybrid scheme, prand=", prand)

                aw2d = np.maximum(convw[0:-1, :], diffw + (1 - fx) * convw[0:-1, :])
                aw2d = np.maximum(aw2d, 0.0)

                ae2d = np.maximum(
                    -convw[1:, :],
                    np.roll(diffw, -1, axis=0) - np.roll(fx, -1, axis=0) * convw[1:, :],
                )
                ae2d = np.maximum(ae2d, 0.0)

                as2d = np.maximum(convs[:, 0:-1], diffs + (1 - fy) * convs[:, 0:-1])
                as2d = np.maximum(as2d, 0.0)

                an2d = np.maximum(
                    -convs[:, 1:],
                    np.roll(diffs, -1, axis=1) - np.roll(fy, -1, axis=1) * convs[:, 1:],
                )
                an2d = np.maximum(an2d, 0.0)
            
            if scheme_local == "c":
                if iter == 0:
                    print("CDS scheme, prand=", prand)
                aw2d = diffw + (1 - fx) * convw[0:-1, :]
                ae2d = np.roll(diffw, -1, axis=0) - np.roll(fx, -1, axis=0) * convw[1:, :]

                as2d = diffs + (1 - fy) * convs[:, 0:-1]
                an2d = np.roll(diffs, -1, axis=1) - np.roll(fy, -1, axis=1) * convs[:, 1:]

            aw2d[0, :] = 0
            ae2d[-1, :] = 0
            as2d[:, 0] = 0
            an2d[:, -1] = 0
            
            return aw2d, ae2d, as2d, an2d, su2d, sp2d


        def bc(
            su2d,
            sp2d,
            phi_bc_west,
            phi_bc_east,
            phi_bc_south,
            phi_bc_north,
            phi_bc_west_type,
            phi_bc_east_type,
            phi_bc_south_type,
            phi_bc_north_type,
        ):
            su2d = np.zeros((ni, nj))
            sp2d = np.zeros((ni, nj))

            # south
            if phi_bc_south_type == "d":
                sp2d[:, 0] = sp2d[:, 0] - viscos * as_bound
                su2d[:, 0] = su2d[:, 0] + viscos * as_bound * phi_bc_south

            # north
            if phi_bc_north_type == "d":
                sp2d[:, -1] = sp2d[:, -1] - viscos * an_bound
                su2d[:, -1] = su2d[:, -1] + viscos * an_bound * phi_bc_north

            # west
            if phi_bc_west_type == "d":
                sp2d[0, :] = sp2d[0, :] - viscos * aw_bound
                su2d[0, :] = su2d[0, :] + viscos * aw_bound * phi_bc_west
            # east
            if phi_bc_east_type == "d":
                sp2d[-1, :] = sp2d[-1, :] - viscos * ae_bound
                su2d[-1, :] = su2d[-1, :] + viscos * ae_bound * phi_bc_east

            return su2d, sp2d


        def conv(u2d, v2d, p2d_face_w, p2d_face_s):
            # compute convection
            u2d_face_w, u2d_face_s = compute_face_phi(
                u2d,
                u_bc_west,
                u_bc_east,
                u_bc_south,
                u_bc_north,
                u_bc_west_type,
                u_bc_east_type,
                u_bc_south_type,
                u_bc_north_type,
            )
            v2d_face_w, v2d_face_s = compute_face_phi(
                v2d,
                v_bc_west,
                v_bc_east,
                v_bc_south,
                v_bc_north,
                v_bc_west_type,
                v_bc_east_type,
                v_bc_south_type,
                v_bc_north_type,
            )

            apw = np.zeros((ni + 1, nj),dtype=precision)
            aps = np.zeros((ni, nj + 1),dtype=precision)

            convw = -u2d_face_w * areawx - v2d_face_w * areawy
            convs = -u2d_face_s * areasx - v2d_face_s * areasy

            # \\\\\\\\\\\\\\\\\ west face

            # create ghost cells at east & west boundaries with Neumann b.c.
            p2d_e = p2d
            p2d_w = p2d
            # duplicate last row and put it at the end

            # duplicate row 0 and put it before row 0 (west boundary)
            if useCupy:
                p2d_e_row = p2d_e[-1.:].copy()
                p2d_e = np.vstack((p2d_e, p2d_e_row))

                p2d_w_row = p2d_w[0,:].copy()
                p2d_w = np.vstack((p2d_w_row, p2d_w))
            else:
                p2d_e = np.insert(p2d_e, -1, p2d_e[-1, :], axis=0)
                p2d_w = np.insert(p2d_w, 0, p2d_w[0, :], axis=0)

            dp = np.roll(p2d_e, -1, axis=0) - 3 * p2d_e + 3 * p2d_w - np.roll(p2d_w, 1, axis=0)

            apw[0:-1, :] = fx * ap2d_vel + (1 - fx) * np.roll(ap2d_vel, 1, axis=0)
            apw[-1, :] = 1e-20

            dvelw = dp * areaw / 4 / apw

            # boundaries (no corrections)
            dvelw[0, :] = 0
            dvelw[-1, :] = 0

            convw = convw + areaw * dvelw

            # \\\\\\\\\\\\\\\\\ south face
            # create ghost cells at north & south boundaries with Neumann b.c.
            p2d_n = p2d
            p2d_s = p2d
            # duplicate last column and put it at the end
            p2d_n_col = p2d_n[:, -1].copy().reshape(-1,1)
            p2d_n = np.hstack((p2d_n_col, p2d_n))

            p2d_s_col =  p2d_s[:, -1].copy().reshape(-1,1)
            p2d_s = np.hstack((p2d_s_col, p2d_s))
            # duplicate first column and put it before column 0 (south boundary)
            dp = np.roll(p2d_n, -1, axis=1) - 3 * p2d_n + 3 * p2d_s - np.roll(p2d_s, 1, axis=1)

            aps[:, 0:-1] = fy * ap2d_vel + (1 - fy) * np.roll(ap2d_vel, 1, axis=1)
            aps[:, -1] = 1e-20

            dvels = dp * areas / 4 / aps

            # boundaries (no corrections)
            dvels[:, 0] = 0
            dvels[:, -1] = 0

            convs = convs + areas * dvels

            # boundaries
            # west
            if u_bc_west_type == "d":
                convw[0, :] = -u_bc_west * areawx[0, :] - v_bc_west * areawy[0, :]
            # east
            if u_bc_east_type == "d":
                convw[-1, :] = -u_bc_east * areawx[-1, :] - v_bc_east * areawy[-1, :]
            # south
            if v_bc_south_type == "d":
                convs[:, 0] = -u_bc_south * areasx[:, 0] - v_bc_south * areasy[:, 0]
            # north
            if v_bc_north_type == "d":
                convs[:, -1] = -u_bc_north * areasx[:, -1] - v_bc_north * areasy[:, -1]

            return convw, convs
        def vist_kom(vis2d, k2d, om2d):
            visold = vis2d
            vis2d = k2d / om2d + viscos

            # modify viscosity
            vis2d = modify_vis(vis2d)

            # under-relax viscosity
            vis2d = urfvis * vis2d + (1.0 - urfvis) * visold

            return vis2d

    # Main functions to accelerate begins here

    def solve_2d(phi2d, aw2d, ae2d, as2d, an2d, su2d, ap2d, tol_conv, nmax, solver_local):
        aw = np.ravel(aw2d)
        ae = np.ravel(ae2d)
        as1 = np.ravel(as2d)
        an = np.ravel(an2d)
        ap = np.ravel(ap2d)
        m = ni * nj

        su = np.ravel(su2d)
        phi = np.ravel(phi2d)

        if useCupy:
            A = cupyx.scipy.sparse.diags(
                [ap, -an[0:-1], -as1[1:], -ae, -aw[nj:]], [0, 1, -1, nj, -nj], format="csr", dtype=precision
            )
        else:
            A = sparse.diags(
                [ap, -an[0:-1], -as1[1:], -ae, -aw[nj:]], [0, 1, -1, nj, -nj], format="csr",dtype=precision
            )

        res_su = np.linalg.norm(su)
        resid_init = np.linalg.norm(A * phi - su)

        phi_org = phi

        if solver_local == "direct":
            if iter == 0:
                print("solver in solve_2d: direct solver")
            info = 0
            resid = np.linalg.norm(A * phi - su)
            phi = linalg.spsolve(A, su)
        if solver_local == "pyamg":
            if iter == 0:
                print("solver in solve_2d: pyamg solver")
            App = pyamg.ruge_stuben_solver(A)  # construct the multigrid hierarchy
            res_amg = []
            phi = App.solve(su, tol=tol_conv, x0=phi, residuals=res_amg)

            info = 0
            print("Residual history in pyAMG", ["%0.4e" % i for i in res_amg])
        if solver_local == "cgs":
            if iter == 0:
                print("solver in solve_2d: cgs")
            phi, info = linalg.cgs(A, su, x0=phi, atol=tol_conv, maxiter=nmax)  # good
        if solver_local == "cg":
            if iter == 0:
                print("solver in solve_2d: cg")
            phi, info = linalg.cg(A, su, x0=phi, atol=tol_conv, maxiter=nmax)  # good
        if solver_local == "gmres":
            if iter == 0:
                print("solver in solve_2d: gmres")
            if useCupy:
                phi, info = cupyx.scipy.sparse.linalg.gmres(A, su, x0=phi, atol=tol_conv, maxiter=nmax)  # good
            else:
                phi, info = scipy.sparse.linalg.gmres(A, su, x0=phi, atol=tol_conv, maxiter=nmax)  # good
        if solver_local == "qmr":
            if iter == 0:
                print("solver in solve_2d: qmr")
            phi, info = linalg.qmr(A, su, x0=phi, atol=tol_conv, maxiter=nmax)  # good
        if solver_local == "lgmres":
            if iter == 0:
                print("solver in solve_2d: lgmres")
            phi, info = linalg.lgmres(A, su, x0=phi, atol=tol_conv, maxiter=nmax)  # good

        ### WORK AREA STUDENT PROJECT ###

        if solver_local == "pyamgx":
            if iter == 0:
                print("solver in solve_2d: pyamgx")

            cfg = pyamgx.Config().create_from_dict(
                {
                    "config_version": 2,
                    "determinism_flag": 1,
                    "exception_handling": 1,
                    "solver": {
                        "monitor_residual": 1,
                        "solver": "BICGSTAB",
                        "convergence": "RELATIVE_INI_CORE",
                        "preconditioner": {
                            "solver": "NOSOLVER"
                        }
                    },
                }
            )

            rsc = pyamgx.Resources()
            rsc.create_simple(cfg)
            solver = pyamgx.Solver().create(rsc, cfg)

            A_dev = pyamgx.Matrix().create(rsc)
            su_dev = pyamgx.Vector().create(rsc)
            x_dev = pyamgx.Vector().create(rsc)

            sol = np.zeros(A.shape[0],dtype=precision)  # check dimensions Ax = b

            A_dev.upload_CSR(A)  # A
            su_dev.upload(su)  # b
            x_dev.upload(phi)  # x = phi
            solver.setup(A_dev)
            solver.solve(su_dev, x_dev)
            x_dev.download(sol)

            phi = sol

            A_dev.destroy()
            x_dev.destroy()
            su_dev.destroy()
            rsc.destroy()

        # This loops over an extreme amount of zeros and will never finnish
        if solver_local == "jacobi_dense":
            if iter == 0:
                print("solver in solve_2d: jacobi (dense) method")
            phi_updated = phi
            it = 0
            info = 1
            while it < nmax:
                for i in range(A.shape[0]):
                    sigma = 0
                    for j in range(A.shape[0]):
                        if j != i:
                            sigma += A[i,j] * phi[j]
                    phi_updated[i] = (su[i] - sigma) / A[i, i]
                phi = phi_updated
                if np.linalg.norm(A.dot(phi)-su) <= tol_conv:
                    it = nmax
                    info = 0
                it += 1 
      
        if solver_local == "jacobi": # Uses sparse params
            if iter == 0:
                print("solver in solve_2d: jacobi (sparse) method")
            iMask = A.indptr
            jMask = A.indices
            vals = A.data
            phi_updated = phi
            it = 0
            info = 1
            while it < nmax:
                i = 0
                while i < iMask.shape[0]-1:
                    sigma = 0
                    Aii = 0
                    for onRow in range(int(iMask[i]), int(iMask[i+1])):
                        j = jMask[onRow]
                        if j != i:
                            sigma += vals[onRow]*phi[j] # vals[onRow] = A[i,j]
                        else:
                            Aii = vals[onRow]
                    if int(iMask[i]) == int(iMask[i+1]):
                        print("Error, missing value on diagonal in sparse matrix")
                        Aii = 9999999

                    phi_updated[i] = (su[i] - sigma) / Aii #A[i, i]
                    if int(iMask[i]) == int(iMask[i+1]):
                        print(phi_updated[i])
                    i += 1
                phi = phi_updated
                if np.linalg.norm(A.dot(phi)-su) <= tol_conv:
                    it = nmax
                    info = 0
                it += 1 

        ### END STUDENT SOLVERS ###

        if info > 0:
            print(
                "warning in module solve_2d: convergence in sparse matrix solver not reached"
            )

        # compute residual without normalizing with |b|=|su2d|
        if solver_local != "direct":
            resid = np.linalg.norm(A * phi - su)

        delta_phi = np.max(np.abs(phi - phi_org))

        phi2d = np.reshape(phi, (ni, nj))
        phi2d_org = np.reshape(phi_org, (ni, nj))
        
        if solver_local != "pyamg":
            print(
                f"{'residual history in solve_2d: initial residual: '} {resid_init:.2e}{'final residual: ':>30}{resid:.2e}\
          {'delta_phi: ':>25}{delta_phi:.2e}"
            )
        
        # we return the initial residual; otherwise the solution is always satisfied (but the non-linearity is not accounted for)
        return phi2d, resid_init


    def calcu(su2d, sp2d, p2d_face_w, p2d_face_s):
        # b.c., sources, coefficients

        # presssure gradient
        dpdx = dphidx(p2d_face_w, p2d_face_s)
        su2d = su2d - dpdx * vol

        # modify su & sp
        su2d, sp2d = modify_u(su2d, sp2d)

        ap2d = aw2d + ae2d + as2d + an2d - sp2d

        # under-relaxation
        ap2d = ap2d / urf_vel
        su2d = su2d + (1 - urf_vel) * ap2d * u2d

        return su2d, sp2d, ap2d


    def calcv(su2d, sp2d, p2d_face_w, p2d_face_s):
        # b.c., sources, coefficients

        # presssure gradient
        dpdy = dphidy(p2d_face_w, p2d_face_s)
        su2d = su2d - dpdy * vol

        # modify su & sp
        su2d, sp2d = modify_v(su2d, sp2d)

        ap2d = aw2d + ae2d + as2d + an2d - sp2d

        # under-relaxation
        ap2d = ap2d / urf_vel
        su2d = su2d + (1 - urf_vel) * ap2d * v2d

        # ap2d will be used in calcp; store it as ap2d_vel
        ap2d_vel = ap2d

        return su2d, sp2d, ap2d, ap2d_vel


    def calck(su2d, sp2d, k2d, om2d, vis2d, u2d_face_w, u2d_face_s, v2d_face_w, v2d_face_s):
        # b.c., sources, coefficients

        # production term
        dudx = dphidx(u2d_face_w, u2d_face_s)
        dvdx = dphidx(v2d_face_w, v2d_face_s)

        dudy = dphidy(u2d_face_w, u2d_face_s)
        dvdy = dphidy(v2d_face_w, v2d_face_s)

        gen = 2.0 * (dudx**2 + dvdy**2) + (dudy + dvdx) ** 2
        vist = np.maximum(vis2d - viscos, 1e-10)
        su2d = su2d + vist * gen * vol

        sp2d = sp2d - cmu * om2d * vol

        # modify su & sp
        su2d, sp2d = modify_k(su2d, sp2d)

        ap2d = aw2d + ae2d + as2d + an2d - sp2d

        # under-relaxation
        ap2d = ap2d / urf_k
        su2d = su2d + (1 - urf_k) * ap2d * k2d

        return su2d, sp2d, gen, ap2d


    def calcom(su2d, sp2d, om2d, gen):
        # --------production term
        su2d = su2d + c_omega_1 * gen * vol

        # --------dissipation term
        sp2d = sp2d - c_omega_2 * om2d * vol

        # modify su & sp
        su2d, sp2d = modify_om(su2d, sp2d)

        ap2d = aw2d + ae2d + as2d + an2d - sp2d

        # under-relaxation
        ap2d = ap2d / urf_vel
        su2d = su2d + (1 - urf_omega) * ap2d * om2d

        return su2d, sp2d, ap2d


    def calcp(pp2d, ap2d_vel):
        # b.c., sources, coefficients and under-relaxation for pp2d

        apw = np.zeros((ni + 1, nj))
        aps = np.zeros((ni, nj + 1))

        pp2d = 0
        # ----------simplec: multiply ap by (1-urf)
        ap2d_vel = np.maximum(ap2d_vel * (1.0 - urf_vel), 1.0e-20)

        # \\\\\\\\\\\\\\\\ west face
        #  visw[0:-1,:,:]=fx*vis_turb+(1-fx)*np.roll(vis_turb,1,axis=0)+viscos
        #  viss[:,0:-1,:]=fy*vis_turb+(1-fy)*np.roll(vis_turb,1,axis=1)+viscos

        #  apw[1:,:]=fx*np.roll(ap2d_vel,-1,axis=0)+(1-fx)*ap2d_vel
        apw[0:-1, :] = fx * ap2d_vel + (1 - fx) * np.roll(ap2d_vel, 1, axis=0)
        apw[0, :] = 1e-20
        dw = areawx**2 + areawy**2
        aw2d = dw[0:-1, :] / apw[0:-1, :]
        ae2d = np.roll(aw2d, -1, axis=0)

        # \\\\\\\\\\\\\\\\ south face
        #  aps[:,1:]=fy*np.roll(ap2d_vel,-1,axis=1)+(1-fy)*ap2d_vel
        aps[:, 0:-1] = fy * ap2d_vel + (1 - fy) * np.roll(ap2d_vel, 1, axis=1)
        aps[:, 0] = 1e-20
        ds = areasx**2 + areasy**2
        as2d = ds[:, 0:-1] / aps[:, 0:-1]
        an2d = np.roll(as2d, -1, axis=1)

        as2d[:, 0] = 0
        an2d[:, -1] = 0
        aw2d[0, :] = 0
        ae2d[-1, :] = 0

        ap2d = aw2d + ae2d + as2d + an2d

        # continuity error
        #  su2d=convw[0:-1,:]-np.roll(convw[0:-1,:],-1,axis=0)+convs[:,0:-1]-np.roll(convs[:,0:-1],-1,axis=1)
        su2d = convw[0:-1, :] - convw[1:, :] + convs[:, 0:-1] - convs[:, 1:]

        # set pp2d=0 in [0,0] tp make it non-singular
        as2d[0, 0] = 0
        an2d[0, 0] = 0
        aw2d[0, 0] = 0
        ae2d[0, 0] = 0
        ap2d[0, 0] = 1
        #  su2d[0,0]=0

        return aw2d, ae2d, as2d, an2d, su2d, ap2d


    def correct_u_v_p(u2d, v2d, p2d):
        # correct convections
        # \\\\\\\\\\\\\ west face
        convw[1:-1, :] = convw[1:-1, :] + aw2d[0:-1, :] * (pp2d[1:, :] - pp2d[0:-1, :])

        # \\\\\\\\\\\\\ south face
        convs[:, 1:-1] = convs[:, 1:-1] + as2d[:, 0:-1] * (pp2d[:, 1:] - pp2d[:, 0:-1])

        # correct p
        p2d = p2d + urf_p * (pp2d - pp2d[0, 0])

        # compute pressure correecion at faces (N.B. p_bc_west,, ... are not used since we impose Neumann b.c., everywhere)
        pp2d_face_w, pp2d_face_s = compute_face_phi(
            pp2d, p_bc_west, p_bc_east, p_bc_south, p_bc_north, "n", "n", "n", "n"
        )

        dppdx = dphidx(pp2d_face_w, pp2d_face_s)
        u2d = u2d - dppdx * vol / ap2d_vel

        dppdy = dphidy(pp2d_face_w, pp2d_face_s)
        v2d = v2d - dppdy * vol / ap2d_vel

        return convw, convs, p2d, u2d, v2d, su2d



    ######################### the execution of the code starts here #############################

    ########### grid specification ###########

    datax = np.loadtxt("meshes/" + mesh + "x2d.dat",dtype=precision)
    x = datax[0:-1]
    ni = int(datax[-1])
    datay = np.loadtxt("meshes/" + mesh + "y2d.dat",dtype=precision)
    y = datay[0:-1]
    nj = int(datay[-1])

    x2d = np.zeros((ni + 1, nj + 1),dtype=precision)
    y2d = np.zeros((ni + 1, nj + 1),dtype=precision)

    x2d = np.reshape(x, (ni + 1, nj + 1))
    y2d = np.reshape(y, (ni + 1, nj + 1))

    # compute cell centers
    xp2d = 0.25 * (x2d[0:-1, 0:-1] + x2d[0:-1, 1:] + x2d[1:, 0:-1] + x2d[1:, 1:])
    yp2d = 0.25 * (y2d[0:-1, 0:-1] + y2d[0:-1, 1:] + y2d[1:, 0:-1] + y2d[1:, 1:])

    # initialize geometric arrays

    vol = np.zeros((ni, nj),dtype=precision)
    areas = np.zeros((ni, nj + 1),dtype=precision)
    areasx = np.zeros((ni, nj + 1),dtype=precision)
    areasy = np.zeros((ni, nj + 1),dtype=precision)
    areaw = np.zeros((ni + 1, nj),dtype=precision)
    areawx = np.zeros((ni + 1, nj),dtype=precision)
    areawy = np.zeros((ni + 1, nj),dtype=precision)
    areaz = np.zeros((ni, nj),dtype=precision)
    as_bound = np.zeros((ni),dtype=precision)
    an_bound = np.zeros((ni),dtype=precision)
    aw_bound = np.zeros((nj),dtype=precision)
    ae_bound = np.zeros((nj),dtype=precision)
    az_bound = np.zeros((ni, nj),dtype=precision)
    fx = np.zeros((ni, nj),dtype=precision)
    fy = np.zeros((ni, nj),dtype=precision)

    setup_case()

    (
        areaw,
        areawx,
        areawy,
        areas,
        areasx,
        areasy,
        vol,
        fx,
        fy,
        aw_bound,
        ae_bound,
        as_bound,
        an_bound,
        dist,
    ) = init()


    # initialization
    u2d = np.ones((ni, nj),dtype=precision) * 1e-20
    v2d = np.ones((ni, nj),dtype=precision) * 1e-20
    p2d = np.ones((ni, nj),dtype=precision) * 1e-20
    pp2d = np.ones((ni, nj),dtype=precision) * 1e-20
    k2d = np.ones((ni, nj),dtype=precision) * 1
    om2d = np.ones((ni, nj),dtype=precision) * 1
    vis2d = np.ones((ni, nj),dtype=precision) * viscos
    
    fmu2d = np.ones((ni, nj),dtype=precision)
    gen = np.ones((ni, nj),dtype=precision)

    convw = np.ones((ni + 1, nj),dtype=precision) * 1e-20
    convs = np.ones((ni, nj + 1),dtype=precision) * 1e-20

    aw2d = np.ones((ni, nj),dtype=precision) * 1e-20
    ae2d = np.ones((ni, nj),dtype=precision) * 1e-20
    as2d = np.ones((ni, nj),dtype=precision) * 1e-20
    an2d = np.ones((ni, nj),dtype=precision) * 1e-20
    al2d = np.ones((ni, nj),dtype=precision) * 1e-20
    ah2d = np.ones((ni, nj),dtype=precision) * 1e-20
    ap2d = np.ones((ni, nj),dtype=precision) * 1e-20
    ap2d_vel = np.ones((ni, nj),dtype=precision) * 1e-20
    su2d = np.ones((ni, nj),dtype=precision) * 1e-20
    sp2d = np.ones((ni, nj),dtype=precision) * 1e-20
    ap2d = np.ones((ni, nj),dtype=precision) * 1e-20
    dudx = np.ones((ni, nj),dtype=precision) * 1e-20
    dudy = np.ones((ni, nj),dtype=precision) * 1e-20
    usynt_inlet = np.ones((nj),dtype=precision) * 1e-20
    vsynt_inlet = np.ones((nj),dtype=precision) * 1e-20
    wsynt_inlet = np.ones((nj),dtype=precision) * 1e-20

    # comute Delta_max for LES/DES/PANS models
    delta_max = np.maximum(vol / areas[:, 1:], vol / areaw[1:, :])

    # initialize
    u2d, v2d, k2d, om2d, vis2d = modify_init(u2d, v2d, k2d, om2d, vis2d)

    k2d = np.maximum(k2d, 1e-6)

    u2d_face_w, u2d_face_s = compute_face_phi(
        u2d,
        u_bc_west,
        u_bc_east,
        u_bc_south,
        u_bc_north,
        u_bc_west_type,
        u_bc_east_type,
        u_bc_south_type,
        u_bc_north_type,
    )
    v2d_face_w, v2d_face_s = compute_face_phi(
        v2d,
        v_bc_west,
        v_bc_east,
        v_bc_south,
        v_bc_north,
        v_bc_west_type,
        v_bc_east_type,
        v_bc_south_type,
        v_bc_north_type,
    )
    p2d_face_w, p2d_face_s = compute_face_phi(
        p2d,
        p_bc_west,
        p_bc_east,
        p_bc_south,
        p_bc_north,
        p_bc_west_type,
        p_bc_east_type,
        p_bc_south_type,
        p_bc_north_type,
    )

    u_bc_west, v_bc_west, k_bc_west, om_bc_west, u2d_face_w, convw = modify_inlet()

    convw, convs = conv(u2d, v2d, p2d_face_w, p2d_face_s)

    if kom:
        urf_temp = urfvis  # no under-relaxation
        urfvis = 1
        vis2d = vist_kom(vis2d, k2d, om2d)
        urfvis = urf_temp

    residual_u = 0
    residual_v = 0
    residual_p = 0
    residual_k = 0
    residual_om = 0

    profiling_data = np.array([])
    profiling_labels = ['iteration', 'IO initial', 'IO $u$', 'Solve $u$', 'IO $v$', 'Solve $v$', 'IO $p$', 'Solve $p$', 'Correct pressure', 'Calculate new faces', 'IO $k$', 'Solve $k$', 'IO $\omega$', 'Solve $\omega$', 'Residual $u$', 'Residual $v$', 'Residual $p$', 'Residual $k$', 'Residual $\omega$','RAM usage','VRAM usage', 'Total time']

    print("Saple of floating point precisions: " , x2d.dtype,convw.dtype,u2d.dtype)

    ######################### start of global iteration process #############################

    for iter in range(0, maxit):
        profiling_temp = [iter]

        start_time_iter = time.time()
        # coefficients for velocities
        start_time = time.time()
        # conpute inlet fluc
        if iter == 0:
            u_bc_west, v_bc_west, k_bc_west, om_bc_west, u2d_face_w, convw = modify_inlet()
        aw2d, ae2d, as2d, an2d, su2d, sp2d = coeff(convw, convs, vis2d, 1, scheme)
        # initial IO
        profiling_temp.append(time.time()-start_time); start_time = time.time()

        # u2d
        # boundary conditions for u2d
        su2d, sp2d = bc(
            su2d,
            sp2d,
            u_bc_west,
            u_bc_east,
            u_bc_south,
            u_bc_north,
            u_bc_west_type,
            u_bc_east_type,
            u_bc_south_type,
            u_bc_north_type,
        )
        su2d, sp2d, ap2d = calcu(su2d, sp2d, p2d_face_w, p2d_face_s)
        # IO u
        profiling_temp.append(time.time()-start_time); start_time = time.time()


        u2d, residual_u = solve_2d(
            u2d,
            aw2d,
            ae2d,
            as2d,
            an2d,
            su2d,
            ap2d,
            convergence_limit_u,
            nsweep_vel,
            solver_vel,
        )
        print(f"{'time u: '}{time.time()-start_time:.2e}")
        # solve u
        profiling_temp.append(time.time()-start_time); start_time = time.time()

        # v2d
        # boundary conditions for v2d
        su2d, sp2d = bc(
            su2d,
            sp2d,
            v_bc_west,
            v_bc_east,
            v_bc_south,
            v_bc_north,
            v_bc_west_type,
            v_bc_east_type,
            v_bc_south_type,
            v_bc_north_type,
        )
        su2d, sp2d, ap2d, ap2d_vel = calcv(su2d, sp2d, p2d_face_w, p2d_face_s)
        # IO v
        profiling_temp.append(time.time()-start_time); start_time = time.time()

        v2d, residual_v = solve_2d(
            v2d,
            aw2d,
            ae2d,
            as2d,
            an2d,
            su2d,
            ap2d,
            convergence_limit_v,
            nsweep_vel,
            solver_vel,
        )
        print(f"{'time v: '}{time.time()-start_time:.2e}")
        # solve v
        profiling_temp.append(time.time()-start_time); start_time = time.time()

        # pp2d
        convw, convs = conv(u2d, v2d, p2d_face_w, p2d_face_s)
        convw = modify_outlet(convw)
        aw2d, ae2d, as2d, an2d, su2d, ap2d = calcp(pp2d, ap2d_vel)
        pp2d = np.zeros((ni, nj))
        # IO p
        profiling_temp.append(time.time()-start_time); start_time = time.time()

        pp2d, dummy = solve_2d(
            pp2d,
            aw2d,
            ae2d,
            as2d,
            an2d,
            su2d,
            ap2d,
            convergence_limit_pp,
            nsweep_pp,
            solver_pp,
        )
        # solve p
        profiling_temp.append(time.time()-start_time); start_time = time.time()

        # correct u, v, w, p
        convw, convs, p2d, u2d, v2d, su2d = correct_u_v_p(u2d, v2d, p2d)
        convw = modify_outlet(convw)

        # continuity error
        su2d = (
            convw[0:-1, :]
            - np.roll(convw[0:-1, :], -1, axis=0)
            + convs[:, 0:-1]
            - np.roll(convs[:, 0:-1], -1, axis=1)
        )
        residual_pp = abs(np.sum(su2d))
        print(f"{'time pp: '}{time.time()-start_time:.2e}")
        # correct pressure
        profiling_temp.append(time.time()-start_time); start_time = time.time()

        u2d_face_w, u2d_face_s = compute_face_phi(
            u2d,
            u_bc_west,
            u_bc_east,
            u_bc_south,
            u_bc_north,
            u_bc_west_type,
            u_bc_east_type,
            u_bc_south_type,
            u_bc_north_type,
        )
        v2d_face_w, v2d_face_s = compute_face_phi(
            v2d,
            v_bc_west,
            v_bc_east,
            v_bc_south,
            v_bc_north,
            v_bc_west_type,
            v_bc_east_type,
            v_bc_south_type,
            v_bc_north_type,
        )
        p2d_face_w, p2d_face_s = compute_face_phi(
            p2d,
            p_bc_west,
            p_bc_east,
            p_bc_south,
            p_bc_north,
            p_bc_west_type,
            p_bc_east_type,
            p_bc_south_type,
            p_bc_north_type,
        )

        # correct calc new faces
        profiling_temp.append(time.time()-start_time); start_time = time.time()

        if kom:
            vis2d = vist_kom(vis2d, k2d, om2d)
            # coefficients
            aw2d, ae2d, as2d, an2d, su2d, sp2d = coeff(
                convw, convs, vis2d, prand_k, scheme_turb
            )
            # k
            # boundary conditions for k2d
            su2d, sp2d = bc(
                su2d,
                sp2d,
                k_bc_west,
                k_bc_east,
                k_bc_south,
                k_bc_north,
                k_bc_west_type,
                k_bc_east_type,
                k_bc_south_type,
                k_bc_north_type,
            )
            su2d, sp2d, gen, ap2d = calck(
                su2d, sp2d, k2d, om2d, vis2d, u2d_face_w, u2d_face_s, v2d_face_w, v2d_face_s
            )

            # IO k
            profiling_temp.append(time.time()-start_time); start_time = time.time()
            k2d, residual_k = solve_2d(
                k2d,
                aw2d,
                ae2d,
                as2d,
                an2d,
                su2d,
                ap2d,
                convergence_limit_k,
                nsweep_kom,
                solver_turb,
            )
            k2d = np.maximum(k2d, 1e-10)
            print(f"{'time k: '}{time.time()-start_time:.2e}")
            # solve k
            profiling_temp.append(time.time()-start_time); start_time = time.time()

            # omega
            # boundary conditions for om2d
            aw2d, ae2d, as2d, an2d, su2d, sp2d = coeff(
                convw, convs, vis2d, prand_omega, scheme_turb
            )
            su2d, sp2d = bc(
                su2d,
                sp2d,
                om_bc_west,
                om_bc_east,
                om_bc_south,
                om_bc_north,
                om_bc_west_type,
                om_bc_east_type,
                om_bc_south_type,
                om_bc_north_type,
            )
            su2d, sp2d, ap2d = calcom(su2d, sp2d, om2d, gen)

            aw2d, ae2d, as2d, an2d, ap2d, su2d, sp2d = fix_omega()
        
            # IO omega
            profiling_temp.append(time.time()-start_time); start_time = time.time()
            om2d, residual_om = solve_2d(
                om2d,
                aw2d,
                ae2d,
                as2d,
                an2d,
                su2d,
                ap2d,
                convergence_limit_om,
                nsweep_kom,
                solver_turb,
            )
            om2d = np.maximum(om2d, 1e-10)

            print(f"{'time omega: '}{time.time()-start_time:.2e}")
        
            # solve omega
            profiling_temp.append(time.time()-start_time); start_time = time.time()

        # scale residuals
        residual_u = residual_u / resnorm_vel
        residual_v = residual_v / resnorm_vel
        residual_p = residual_p / resnorm_p
        residual_k = residual_k / resnorm_vel**2
        residual_om = residual_om / resnorm_vel

        resmax = max(residual_u, residual_v, residual_p)
        
        print(
            f"\n{'--iter:'}{iter:d}, {'max residual:'}{resmax:.2e}, {'u:'}{residual_u:.2e}\
    , {'v:'}{residual_v:.2e}, {'pp:'}{residual_pp:.2e}, {'k:'}{residual_k:.2e}\
    , {'om:'}{residual_om:.2e}\n"
        )

        print(
            f"\n{'monitor iteration:'}{iter:4d}, {'u:'}{u2d[imon,jmon]: .2e}\
    , {'v:'}{v2d[imon,jmon]: .2e}, {'p:'}{p2d[imon,jmon]: .2e}\
    , {'k:'}{k2d[imon,jmon]: .2e}, {'om:'}{om2d[imon,jmon]: .2e}\n"
        )
        
        vismax = np.max(vis2d.flatten()) / viscos
        umax = np.max(u2d.flatten())
        ommin = np.min(om2d.flatten())
        kmin = np.min(k2d.flatten())
        
        print(
            f"\n{'---iter: '}{iter:2d}, {'umax: '}{umax:.2e},{'vismax: '}{vismax:.2e}, {'kmin: '}{kmin:.2e}, {'ommin: '}{ommin:.2e}\n"
        )
       
        print(f"{'time one iteration: '}{time.time()-start_time_iter:.2e}")

        vram_usage = 0 # Default to zero to be compatible with non-GPU systems
        if useCupy or usePyamgx:
            # Get handle for the first GPU
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Get memory information
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            print(f"Total VRAM: {mem_info.total / 1024**2} MB")
            print(f"Used VRAM: {mem_info.used / 1024**2} MB")
            print(f"Free VRAM: {mem_info.free / 1024**2} MB")

            vram_usage = mem_info.used / 1024**2#torch.cuda.memory_allocated(0)*1024 # 0#mem_info.free / 1024**2
        ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        print(f"Current RAM usage {ram_usage}")
        print(f"Current vRAM usage {vram_usage}")

       
        profiling_temp.append(float(residual_u))
        profiling_temp.append(float(residual_v))
        profiling_temp.append(float(residual_p))
        profiling_temp.append(float(residual_k))
        profiling_temp.append(float(residual_om)) 
        profiling_temp.append(float(ram_usage)) 
        profiling_temp.append(float(vram_usage)) 
        profiling_temp.append(time.time()-start_time_iter)
        if iter == 0:
            profiling_data = np.append(profiling_data,np.array(profiling_temp))
        else:
            profiling_data = np.vstack((profiling_data,np.array(profiling_temp)))

        if resmax < sormax:
            print(profiling_data)
            #break # Uncommenting makes total # iterations hard to predict

    ######################### end of global iteration process #############################

    if usePyamgx:
        pyamgx.finalize()

    print("program reached normal stop")

    if not dry_run:

        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 12})
        plt.rcParams["mathtext.fontset"] = "stix"
        plt.rcParams["font.family"] = "STIXGeneral"
        plt.rcParams["figure.figsize"] = (3.2, 2.4)
        plt.rcParams["savefig.bbox"] = "tight"

        if useCupy:
            x2d = np.asnumpy(x2d)
            y2d = np.asnumpy(y2d)
            u2d = np.asnumpy(u2d)
            v2d = np.asnumpy(v2d)
            k2d = np.asnumpy(k2d)
            om2d = np.asnumpy(om2d)
            p2d = np.asnumpy(p2d)
            vis2d = np.asnumpy(vis2d)
            
            xp2d = np.asnumpy(xp2d)
            yp2d = np.asnumpy(yp2d)

            profiling_data = np.asnumpy(profiling_data) 
            # swap back to np-package
            import numpy as np
            
        dataset = profiling_data
        datatype = conf["precision"] 
        timeStamp = time.strftime("%Y-%m-%d-%H%M%S", time.gmtime()) 
        run_tag = f"figures/{mesh}_{str(maxit)}_{datatype}_cupy{useCupy}_vel_{solver_vel}_pp_{solver_pp}_turb_{solver_turb}_{timeStamp}"
        # note, can't use spaces in names
        os.system('mkdir ' + run_tag)
        colors = "coolwarm"

        with open(run_tag + "/config.json", "w") as fp:
            json.dump(conf , fp)

        ymax = y2d.max()/500 # Boundary layer flow, just look at boundary
        ###

        def plot2d (arr2d,name):
            fig = plt.figure()
            conplot = plt.contourf(x2d[:-1,:-1],y2d[:-1,:-1],arr2d, cmap=colors,levels=10)
            plt.xlabel("$x$ (m)")
            plt.ylabel("$y$ (m)")
            plt.ylim(top=ymax)
            ax = plt.gca()
            cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
            cbar = plt.colorbar(conplot,cax=cax)
            cbar.set_label(name)
            plt.savefig(run_tag + "/boundary-"+ name +".pdf", format='pdf',bbox_inches='tight',dpi=800)
            plt.clf()
            #   
            fig = plt.figure()
            conplot = plt.contourf(x2d[:-1,:-1],y2d[:-1,:-1],u2d, cmap=colors,levels=10)
            plt.xlabel("$x$ (m)")
            plt.ylabel("$y$ (m)")
            ax = plt.gca()
            #ax.set_aspect('equal', 'box')
            cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
            cbar = plt.colorbar(conplot,cax=cax)
            cbar.set_label(name)
            plt.savefig(run_tag + "/full_field" + name + ".pdf", format='pdf',bbox_inches='tight',dpi=800)
            plt.clf()
            #   

        plot2d(u2d,"$u$ (ms$^{-1}$)")
        plot2d(v2d,"$v$ (ms$^{-1}$)")
        plot2d(p2d,"$p$ (Pa)")
        plot2d(k2d,"$k$ (m$^2$s$^{-2}$)")
        plot2d(om2d,"$\omega$ (s$^{-1}$)")
        plot2d(vis2d,"$\\nu_T$ (m$^2$s$^{-1}$)")

        cum_time_params = np.array([0]) # to match index of label_array
        for i in range(1,dataset.shape[1]):
            current_dataseries = np.abs(dataset[:,i]) + 1E-10
            plt.plot(dataset[:,0],current_dataseries)
            #ax = plt.axes()
            plt.xlabel(profiling_labels[0] + " (-)")
            plt.ylabel(profiling_labels[i] + " (s)")
            if "Residual" in str(profiling_labels[i]):
                plt.ylabel(profiling_labels[i] + " (-)")
                plt.yscale('log')
            if "usage" in str(profiling_labels[i]):
                plt.ylabel(profiling_labels[i] + " (MB)")
            plt.savefig(run_tag + "/" + str(profiling_labels[i]) + ".pdf", format='pdf',bbox_inches='tight',dpi=800)
            plt.clf()
            cum_time_params = np.append(cum_time_params,np.sum(current_dataseries))

        np.save(run_tag + "/cumtime.csv", cum_time_params[1:-8])
        np.save(run_tag + "/cumtime_labels.csv", profiling_labels[1:-8])
        np.save(run_tag + "/main_data.csv", profiling_data) 
        np.save(run_tag + "/main_data_labels.csv", profiling_labels)

        fig = plt.gcf()
        fig.set_size_inches(6,5)
        fig.suptitle(f'Total time {np.sum(dataset[:,-1]): .2f} seconds')
        plt.pie(cum_time_params[1:-8],labels=profiling_labels[1:-8],autopct= lambda x: '{:.0f}%'.format(x))
        plt.savefig(run_tag + "/pie_comparision.pdf", format='pdf',bbox_inches='tight',dpi=800)
        plt.clf() 

        fig = plt.gcf()
        fig.set_size_inches(6,5)
        fig.suptitle(f'Total time exluding pressure {np.sum(dataset[:,-1])-cum_time_params[6]: .2f} seconds')
        noPforPie = np.delete(cum_time_params[1:-8],6)
        noPforPieLabels = np.delete(profiling_labels[1:-8],6)
        plt.pie(noPforPie,labels=noPforPieLabels,autopct= lambda x: '{:.0f}%'.format(x))
        plt.savefig(run_tag + "/pie_comparision_no_p.pdf", format='pdf',bbox_inches='tight',dpi=800)
        plt.clf()

        fig = plt.gcf()
        fig.set_size_inches(6,5)
        fig.suptitle(f'Total time, exluding $p$, {np.sum(dataset[:,-1]-cum_time_params[6]): .2f} seconds')
        cum_time_params[6] = 0
        plt.pie(cum_time_params[1:-8],labels=profiling_labels[1:-8],autopct= lambda x: '{:.0f}%'.format(x))
        plt.savefig(run_tag + "/pie_comparision_noP_co_colors.pdf", format='pdf',bbox_inches='tight',dpi=800)
        plt.clf() 
        plt.close()
