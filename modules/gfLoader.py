#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
import os

def main():
    # Define the velocity model in meters and m/s, and kg/m^3
    thickness = np.array([20.0, 25.0, 75.0, 180.0, 700.0, 4000.0, 0.0]) # units in meters
    vS = np.array([300.0, 500.0, 1500.0, 2500.0, 3000.0, 3360.0, 3750.0]) # units in m/s
    vP = np.array([1000, 2000.0, 3000.0, 4500.0, 5000.0, 5800.0, 6000.0]) # units in m/s
    rho = np.array([1743.0, 2073.1, 2294.3, 2539.0, 2606.8, 2720.0, 2800.0]) # units in kg/m^3
    fMin = 2.0; fMax = 8.0; # units in Hz
    lambdaFrac = 1/3; # fraction
    lambdaRes = 6; #must be greater than 4
    xMax = 5000.0;# maximum horizontal offset upto which displacements will be used
    zMax = 5000.0; # maximum depth upto which displacements will be used
    maxRec = 500; # same value of number of receivers that qseis can handle in one go check qsglobal.h
    # specify the path to the template file, should have read access
    fTemplate = "/data/gravwav/koley/qseisN/fomosto-qseis/doc/examples/qs6inpNTemp.dat"
    # specify the folder where you want to write all input files, should have rw access
    fInpPath = "/data/gravwav/koley/QseisInp/"
    # path to the qseis binary
    qseisExe = "/data/gravwav/koley/qseisN/fomosto-qseis/src/fomosto_qseis2006b"
    splitFileName = fInpPath + 'splitAll.mat';
    splitMat = loadmat(splitFileName);
    splitAll = splitMat['splitAll'];

    # just returns the file paths, but does not include the file names
    gfInfo = getGFFilesForDepth(splitAll,3,0,5000,fInpPath)

    # file names to be added
    components = ['fh-2.tz', 'fh-2.tr', 'fh-2.tt', 'fz-2.tz', 'fz-2.tr']

    # generate full path to the files, note that depth is important here, if there are two depths, specify correct one
    full_files, runs_info = getGFFileListForDepth(gfInfo, depth=0.0, components=components)

    # load all five components 
    xxW, tVec, distVec = loadGFDepth(full_files, runs_info, minVel=100.0)
    #print(xxW.shape, tVec.shape, distVec.shape)

def getInterpolatedGF(splitAll, zTarget, xMin, xMax, basePath, components, minVel=100.0):
    """
    Load Green's functions for an arbitrary receiver depth zTarget.

    If zTarget exactly matches one of the available depths, the corresponding
    Green's functions are loaded directly. Otherwise, the function interpolates
    between the nearest lower and upper depth Green's functions.

    Parameters
    ----------
    splitAll : np.ndarray
        Matrix returned by Qseis input preparation (contains run info)
    zTarget : float
        Target receiver depth in meters
    xMin, xMax : float
        Minimum and maximum horizontal offsets to include (meters)
    basePath : str
        Base directory where all Qseis run folders are located
    components : list[str]
        List of GF component filenames, e.g.
        ['fh-2.tz', 'fh-2.tr', 'fh-2.tt', 'fz-2.tz', 'fz-2.tr']
    minVel : float, optional
        Minimum velocity threshold used by loadGFDepth (default = 100.0)

    Returns
    -------
    xxW : np.ndarray
        Green's function cube of shape (nt, nx, ncomp)
    tVec : np.ndarray
        Time vector
    distVec : np.ndarray
        Horizontal distance vector (meters)
    """

    # --- Step 1: find available GF depths ---
    gfInfo = getGFFilesForDepth(splitAll, zTarget, xMin, xMax, basePath)
    depths = gfInfo["depths"]

    # --- Step 2: if exact match, just load directly ---
    if len(depths) == 1:
        zMatch = depths[0]
        #print(f"[gfLoader] Exact match found for depth {zMatch:.1f} m")
        files, runs = getGFFileListForDepth(gfInfo, depth=zMatch, components=components)
        xxW, tVec, distVec = loadGFDepth(files, runs, minVel=minVel)
        return xxW, tVec, distVec

    # --- Step 3: otherwise interpolate between lower and upper depth ---
    zLower, zUpper = depths
    #print(f"[gfLoader] Interpolating between depths {zLower:.1f} m and {zUpper:.1f} m for target {zTarget:.1f} m")

    filesL, runsL = getGFFileListForDepth(gfInfo, depth=zLower, components=components)
    filesU, runsU = getGFFileListForDepth(gfInfo, depth=zUpper, components=components)

    xxL, tL, distL = loadGFDepth(filesL, runsL, minVel=minVel)
    xxU, tU, distU = loadGFDepth(filesU, runsU, minVel=minVel)

    # --- Step 4: ensure identical time sampling ---
    if not np.allclose(tL, tU):
        raise ValueError("Time vectors differ between depth layers; cannot interpolate safely")

    # --- Step 5: align to a common distance grid (use the coarser one) ---
    if len(distL) < len(distU):
        distTarget = distL
        xxU = intpGFDistance(xxU, distU, distTarget)
    elif len(distU) < len(distL):
        distTarget = distU
        xxL = intpGFDistance(xxL, distL, distTarget)
    else:
        distTarget = distL

    # --- Step 6: depth interpolation ---
    wZ = (zTarget - zLower) / (zUpper - zLower)
    xxW = (1.0 - wZ) * xxL + wZ * xxU

    #print(f"[gfLoader] Completed depth interpolation (weight={wZ:.3f})")

    return xxW, tL, distTarget


def intpGFDistance(xxW, oldDist, newDist):
    """
    Interpolate Green's function cube xxW (nt x nx x ncomp) from oldDist -> newDist.

    Parameters
    ----------
    xxW : np.ndarray
        Original GF array of shape (nt, nx, ncomp)
    oldDist : np.ndarray
        Original horizontal distance vector
    newDist : np.ndarray
        Target distance vector to interpolate onto

    Returns
    -------
    xxNew : np.ndarray
        Interpolated GF array with shape (nt, len(newDist), ncomp)
    """
    nt, _, ncomp = xxW.shape
    xxNew = np.zeros((nt, len(newDist), ncomp), dtype=xxW.dtype)

    for ic in range(ncomp):
        f = interp1d(oldDist, xxW[:, :, ic], kind="linear",
                     axis=1, bounds_error=False, fill_value=0.0)
        xxNew[:, :, ic] = f(newDist)

    return xxNew
    
def loadGFDepth(files, run_info, minVel=100.0):
    """
    Load and concatenate GF files for one depth, including tapering.

    Parameters
    ----------
    files : list of str
        Full paths to GF files (all runs for all components), ordered as:
        [run0_comp0, run0_comp1, ..., run1_comp0, run1_comp1, ...]
    run_info : list of dict
        List of run info dictionaries for each run, each containing:
        - 'distVec', 'nx', 'x_start', 'x_end'.
    minVel : float, optional
        Minimum velocity for taper window.

    Returns
    -------
    xxW : np.ndarray
        GF matrix of shape (nt, nx_total, ncomp)
    tVec : np.ndarray
        Time vector
    distVec : np.ndarray
        Horizontal positions of all concatenated runs
    """
    n_runs = len(run_info)
    n_comp = int(len(files) / n_runs)  # number of components per run
    xx_list = []
    dist_list = []

    tVec = None

    for i_run, run in enumerate(run_info):
        # Distances for this run
        distVec_run = run['distVec']
        dist_list.append(distVec_run)

        # Container for all components of this run
        run_comps = []

        for i_comp in range(n_comp):
            file_idx = i_run * n_comp + i_comp
            file_path = files[file_idx]

            # Load GF using getGF, applying taper
            xxW_run, tVec_run, _ = getGF(file_path, distVec_run, minVel=minVel)

            if tVec is None:
                tVec = tVec_run
            else:
                # Ensure all time vectors match
                assert np.allclose(tVec, tVec_run), "Time vectors do not match across runs"

            run_comps.append(xxW_run)

        # Stack components along last axis
        run_comps = np.stack(run_comps, axis=-1)  # shape: nt x nx_run x ncomp
        xx_list.append(run_comps)

    # Concatenate all runs along x
    xxW = np.concatenate(xx_list, axis=1)
    distVec = np.concatenate(dist_list)

    return xxW, tVec, distVec
    
def getGFFileListForDepth(gfInfo, depth, components):
    """
    Generate full GF filenames for a given depth and component types.

    Parameters
    ----------
    gfInfo : dict
        Output of getGFFilesForDepth.
    depth : float
        Depth of interest.
    components : list of str
        Component filenames, e.g., ['fh-2.tz', 'fh-2.tr', 'fh-2.tt', 'fz-2.tz', 'fz-2.tr'].

    Returns
    -------
    list of str
        Full paths to all GF files for this depth and the specified components.
    list of dict
        Corresponding run info (distVec, nx, etc.) for each run.
    """
    # Find index for this depth in gfInfo
    try:
        depth_idx = gfInfo['depths'].index(depth)
    except ValueError:
        raise ValueError(f"Depth {depth} not found in gfInfo['depths']")
    
    files = gfInfo['files'][depth_idx]
    runs = gfInfo['runs'][depth_idx]

    full_files = []
    run_info = []

    for f, r in zip(files, runs):
        full_files.extend([os.path.join(f, comp) for comp in components])
        run_info.append(r)

    return full_files, run_info
    
def getGF(fPath, distVec, minVel=100.0):
    """
    Load a Green's function ASCII file and apply distance-dependent tapering.

    Parameters
    ----------
    fPath : str
        Folder containing GF file
    fName : str
        Name of GF file
    distVec : array_like
        Distance vector (meters) corresponding to each column in the GF file
    minVel : float
        Minimum velocity for taper window (m/s)

    Returns
    -------
    xxW : np.ndarray
        Windowed displacement data (time x distance)
    tVec : np.ndarray
        Time vector
    fSamp : float
        Sampling frequency
    """
    # Load the entire file
    data = np.loadtxt(f"{fPath}", skiprows=1)

    tVec = data[:, 0]
    xx = data[:, 1:]  # remove time column

    dt = tVec[1] - tVec[0]
    fSamp = 1.0 / dt

    # Apply distance-dependent taper
    W = genWin(tVec, distVec, minVel)
    xxW = xx*W  # element-wise multiplication

    return xxW, tVec, fSamp


def genWin(tVec, distVec, minVel=100.0):
    """
    Generate distance-dependent taper window.

    Parameters
    ----------
    tVec : array_like
        Time vector (seconds)
    distVec : array_like
        Distance vector (meters)
    minVel : float
        Minimum velocity for taper (m/s)

    Returns
    -------
    W : np.ndarray
        Window matrix (time x distance)
    """
    tVec = np.asarray(tVec)
    distVec = np.asarray(distVec)

    dt = tVec[1] - tVec[0]
    lTVec = len(tVec)
    lDVec = len(distVec)

    # Compute start index for taper
    tMin = distVec / minVel
    tInd = np.ceil(tMin / dt).astype(int)
    wLen = lTVec - tInd

    W = np.ones((lTVec, lDVec))

    for i in range(lDVec):
        if wLen[i] > 3:
            totWin = 2 * wLen[i]
            ww = np.hanning(int(totWin))
            wwHalf = ww[int(totWin//2):]
            W[tInd[i]:, i] = wwHalf

    return W
    
def getGFFilesForDepth(splitAll, zTar, xMin, xMax, basePath):
    """
    Determine which Green's function files need to be loaded for a target depth
    and horizontal extent.

    Returns a dict with:
      'depths': list of 1 or 2 depths,
      'files': list-of-lists of filenames per depth,
      'runs' : list-of-lists of run metadata dicts per depth
    """
    depths_unique = np.unique(splitAll[:, 2])

    # bounds check
    if zTar < depths_unique.min() or zTar > depths_unique.max():
        raise ValueError(f"zTar {zTar} outside database depths "
                         f"[{depths_unique.min()}, {depths_unique.max()}]")

    # Determine target depths (handle exact match)
    # use isclose for robustness
    close_mask = np.isclose(depths_unique, zTar)
    if close_mask.any():
        targetDepths = [depths_unique[np.argmax(close_mask)]]
    else:
        zLower = np.max(depths_unique[depths_unique < zTar])
        zUpper = np.min(depths_unique[depths_unique > zTar])
        targetDepths = [zLower, zUpper]

    filesPerDepth = []
    runsPerDepth = []

    for z in targetDepths:
        runs = splitAll[splitAll[:, 2] == z]

        # collect runs with overlap and their meta
        filteredRuns = []
        runMeta = []
        for run in runs:
            runXStart = float(run[3])
            runXEnd = float(run[4])
            nx = int(run[5])

            # overlap test
            if runXEnd >= xMin and runXStart <= xMax:
                runIndex = int(run[1])
                layerIndex = int(run[0])
                filename = os.path.join(basePath, f"Layer{layerIndex}Run{runIndex}")

                distVec = np.linspace(runXStart, runXEnd, nx)

                filteredRuns.append(filename)
                runMeta.append({
                    'x_start': runXStart,
                    'x_end': runXEnd,
                    'nx': nx,
                    'distVec': distVec
                })

        # sort runs by x_start so concatenation is ordered left→right
        if runMeta:
            order = np.argsort([r['x_start'] for r in runMeta])
            filteredRuns = [filteredRuns[i] for i in order]
            runMeta = [runMeta[i] for i in order]

        filesPerDepth.append(filteredRuns)
        runsPerDepth.append(runMeta)

    return {
        'depths': targetDepths,
        'files': filesPerDepth,
        'runs': runsPerDepth
    }

if __name__ == "__main__":
    main()