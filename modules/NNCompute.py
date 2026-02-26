#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.io import loadmat
import scipy.constants as const
from pysurf96 import surf96
from modules import NNGeometry
from modules import simDisp
from modules import plotUtils

def main():
    # don't use config yet, just hard code for now, can be implemented anytime
    # test the calculation engine first with just a P-wave propagating along +x
    # Define the velocity model in km and km/s
    thickness = np.array([10000.0, 0.0]) # units in meters
    vs = np.array([2000, 2500.0]) # units in m/s
    vp = np.array([4000, 5000.0]) # units in m/s
    rho = np.array([2465.0,2606.0]) # units in kg/m^3
    rhoAir = 1.2; # density of air
    fMin = 2.0; fMax = 8.0; df = 0.05;# units in Hz
    lambdaFrac = 1/3; # fraction
    lambdaRes = 6; #must be greater than 4
    xMaxGF = 5000.0;# maximum horizontal offset upto which displacements will be used
    zMaxGF = 5000.0; # maximum depth upto which displacements will be used
    maxRec = 500; # same value of number of receivers that qseis can handle in one go check qsglobal.h
    
    xMin = -2000.0; xMax = 2000.0; # minimum and maximum of the simulation domain in X-direction (EW)
    yMin = -2000.0; yMax = 2000.0; # maximum and minimum of the simulation domain in Y-direction (NS)
    zMin = 0.0; zMax = 4000.0; # maximum and minimum of the simulation domain in Z-direction (depth)
    domXYBounds = (xMin,xMax,yMin,yMax);
    cubeC = 250.0; rCavity = 20.0;
    cubeS = 2*rCavity;
    cubeTop = cubeC-cubeS; cubeBot = cubeC+cubeS;
    
    G = const.G
    # some other inputs for simDisp
    maxRec = 500; # same value of number of receivers that qseis can handle in one go check qsglobal.h
    tMax = 40; nSamp = 2048;
    # specify the folder where you want to write all input files, should have rw access
    fInpPath = "/data/gravwav/koley/QseisInpN/"
    #fInpPath = "/data/gravwav/koley/SALVUSOut/"
    outDispPath = "/data/gravwav/koley/OutDisp/"
    outDispRea = "/data/gravwav/koley/OutDispRea/"
    components = ['fh-2.tz', 'fh-2.tr', 'fh-2.tt', 'fz-2.tz', 'fz-2.tr']

    # splitAll is necessary so load it
    nRea = 10; # number of realizations
    splitFileName = fInpPath + 'splitAll.mat';
    splitMat = loadmat(splitFileName);
    splitAll = splitMat['splitAll'];

    R1 = 0; R2 = 2500;
    nSrc = 100;
    srcDistri = "internal"
    # frequency axis to be used by simDisp
    freqOut, idxFreq, df_native = simDisp.getFreqGrid(tMax, nSamp, fMin, fMax, df);
    nFreq = len(freqOut)

    # two depth points for scaling, displacement saved per realization
    zList = [0.0,250.0]
    
    # just make sure zMax never coincides with an actual horizontal interface
    # bug to be fixed later if needed
    thetaW = 90; phiW = 0; # theta and phi for a plane wave travelling along +x

    # generate the grid size required
    gridSize = getGridSize(thickness,vp, vs, rho, fMin, fMax, lambdaFrac, lambdaRes);
    print('GridSize = ' + str(gridSize[0]))
    
    print('grid size in each layer = ' + str(gridSize));
    layers = []

    # Create and add multiple layers
    lenThick = len(thickness);
    depths = np.cumsum(thickness[0:(lenThick-1)]);
    depths = np.insert(depths,0,0.0,axis=0);
    
    # find the index to insert zMax
    zMaxInd = np.where(depths<zMax)[0]
    
    newDepths = np.append(depths[0:(zMaxInd[-1]+1)],zMax);
    lenNewDepth = len(newDepths);
    #print(newDepths);
    for i in range(0,(lenNewDepth-1)):
        layers.append(NNGeometry.Layer(xMin=xMin, xMax=xMax, yMin= yMin, yMax = yMax, zTop=newDepths[i], zBot=newDepths[i+1], vP=vp[i], vS=vs[i], rho=rho[i]));

    print('Total number of layers = ' + str(len(layers)));
    nLayers = len(layers)
    
    for reaNo in range(0,nRea):
        print('Doing realization = ' +str(reaNo))
        # get the source distribution per realization
        xSrc, ySrc, azSrc, phiSrc, ampSrc =  simDisp.genAmbSrc(nSrc , mode = srcDistri, R1 = R1, R2 = R2, xMin=xMin, xMax=xMax, yMin=yMin,
                                           yMax=yMax, randomPhase=True, freqDependent=True, nFreq=nFreq)

        # create and save the displacement field on surface and depth for scaling afterwards
        simDisp.getSurfDeepDispPerRea(zList, xSrc, ySrc, azSrc, phiSrc, ampSrc, xMaxGF, splitAll, fMin, fMax, outDispRea,
                    fInpPath, components, reaNo, idxFreq = idxFreq, freqOut=freqOut)
        # preallocate per realization
        IBlockTot = np.zeros((nFreq,3),dtype=np.complex128)
        IVertFaceTot  = np.zeros((nFreq,3),dtype=np.complex128)
        IHorFaceTot = np.zeros((nFreq,3),dtype=np.complex128)
        ITot = np.zeros((nFreq,3),dtype=np.complex128)
        
        for layerNo, layer in enumerate(layers):
            layer = layers[layerNo]

            layer.updateCubeInteraction(cubeTop, cubeBot)
            layer.generateBlocks(xMin, xMax, yMin, yMax, cubeC, cubeS, domXYBounds)

            #freqOut = np.linspace(fMin,fMax,60);
            #nFreq = len(freqOut)
            
            print(layer)
            blkNo = 1
            for blk in layer.blocks:
                #print(blk);
                print('Block = '+str(blkNo))
                print('Integrating volume  NN block in layer' + str(layerNo))
        
                IBlock = runVolNNComputeBlock(blk, gridSize[layerNo], freqOut, cubeC, rCavity, xSrc, ySrc, azSrc,
                                              phiSrc, ampSrc, idxFreq, fMin, fMax, outDispPath, splitAll, xMaxGF,
                                              fInpPath, components, useSimDisp=True, nCPU=4, nChunk=20000,
                                              thetaW=thetaW, phiW=phiW)

                IBlockTot = IBlockTot + IBlock;

                # now compute the surface NN from the vertical faces of the block which form the outer boundary
                for f in blk.verticalFaces:
                    if(f.isBoundary):
                        print('Integrating vertical surface NN block in layer' + str(layerNo))
                        print('vertical boundary face found')
                        # is an outer boundary
                        IVertFace = runVertSurfNNCompute(f, gridSize[layerNo], freqOut, cubeC, xSrc, ySrc, azSrc,
                                                         phiSrc, ampSrc, idxFreq, fMin, fMax, outDispPath, splitAll,
                                                         xMaxGF, fInpPath, components, nCPUDisp = nCPUDisp, nCPUNN=nCPUNN,
                                                         nChunk = 20000, useSimDisp=True, vP_for_test=blk.vP, thetaW=thetaW, phiW=phiW)
                        
                        IVertFaceTot = IVertFaceTot + IVertFace
        
                
                # now compute the surface NN from the horizontal interfaces at the very end of layers
                print('Integrating horizontal surface NN block in layer' + str(layerNo))
                IHorFace = handleHorSurfNN(layer, layers, blk, gridSize[layerNo], nLayers, layerNo, rhoAir, freqOut, 
                                           cubeC, xSrc, ySrc, azSrc, phiSrc, ampSrc, idxFreq, fMin, fMax, outDispPath,
                                           splitAll, xMaxGF, fInpPath, components, nCPU = 4, useSimDisp=True,
                                           vP_for_test=blk.vP, thetaW=thetaW, phiW=phiW, chunk_size=5000, rCavity=rCavity)
                
                IHorFaceTot = IHorFaceTot + IHorFace
                
                blkNo = blkNo+1
    
        IVertFaceTot = IVertFaceTot*G
        IHorFaceTot = IHorFaceTot*G
        IBlockTot = IBlockTot*G
        ITot = (IBlockTot-IVertFaceTot-IHorFaceTot)
        # save the NN per realization
        nnFName = 'NNFullRea' + str(reaNo) + '.npz'
        np.savez(os.path.join(outDispRea, nnFName), IVertFaceTot = IVertFaceTot, IHorFaceTot = IHorFaceTot,
                 IBlockTot = IBlockTot, ITot = ITot, freqOut=freqOut)

    dispPointAllRea, attnAllRea, freqDisp = simDisp.assembleSurfDeepDispAllRea(outDispRea, zList, freqOut, nRea)
    nnAllRea, fNN = assembleNNAllRea(outDispRea,freqOut, nRea)

    # scale NN such that the surface displacement is 1
    nnAllRea = nnAllRea/dispPointAllRea[:,:,0]
    
    figASD, axASD = plotUtils.plotPSDDeepSurfMulti(freqOut, dispPointAllRea[:,:,0], 'Surface', 'b', fig=None, axs=None, 
                                         quantity="ASD")
    figASD, axASD = plotUtils.plotPSDDeepSurfMulti(freqOut, dispPointAllRea[:,:,1], 'Deep', 'r', fig=figASD, axs=axASD, 
                                         quantity="ASD")

    # plot the surface to deep attenuation
    figAttn, axAttn = plotUtils.plotPSDDeepSurfMulti(freqOut, attnAllRea, 'Surf-Deep', 'b', fig=None, axs=None, 
                                         quantity="Attn")
    # load the Terziet attenuation model
    terzAttn = loadmat('/data/gravwav/koley/TerzModel/attnModel.mat');
    attnZ = terzAttn['attnZ']
    attnE = terzAttn['attnE']
    attnN = terzAttn['attnN']

    axAttn[0].plot(attnZ[:,0],attnZ[:,2],'r')
    axAttn[1].plot(attnZ[:,0],attnE[:,2],'r')
    axAttn[2].plot(attnZ[:,0],attnN[:,2],'r')

    # plot the NN acceleration ASD
    figNN, axNN = plotUtils.plotPSDDeepSurfMulti(fNN, nnAllRea, 'NNFull', 'b', fig=None, axs=None, 
                                         quantity="NN acceleration")
    
    fig,axs = plt.subplots(1, 3, figsize=(9,3))
    axs[0].plot(freqOut,np.abs(ITot[:,0]),'b', label = "Simulated")
    axs[0].plot(freqOut,np.abs(IBlockTot[:,0]),'m', label = "Simulated") 
    axs[0].plot(freqOut,np.abs(IVertFaceTot[:,0]),'r', label = "Simulated")
    axs[0].plot(freqOut,np.abs(IHorFaceTot[:,0]),'k', label = "Simulated")
    axs[0].plot(freqOut,np.ones((nFreq,))*8*np.pi/3*G*2800,'g',label = 'Theoretical')
    
    #axs[0].plot(fJ,j0R,'r', label = "Theoretical J_0(2*pi*f*d/v_R(f))");
    axs[0].set_xlim([0,8]);
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('NN_x');
    axs[0].set_title('X component');
    
    axs[1].plot(freqOut,np.abs(ITot[:,1]),'b', label = "Simulated")
    axs[1].plot(freqOut,np.abs(IBlockTot[:,1]),'m', label = "Simulated") 
    axs[1].plot(freqOut,np.abs(IVertFaceTot[:,1]),'r', label = "Simulated")
    axs[1].plot(freqOut,np.abs(IHorFaceTot[:,1]),'k', label = "Simulated")
    axs[1].plot(freqOut,np.ones((nFreq,))*8*np.pi/3*G*2800,'g',label = 'Theoretical')
    
    axs[1].set_xlim([0,8]);
    axs[1].set_ylabel('NN_y');
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_title('Y component');
    
    axs[2].plot(freqOut,np.abs(ITot[:,2]),'b', label = "Simulated")
    axs[2].plot(freqOut,np.abs(IBlockTot[:,2]),'m', label = "Simulated") 
    axs[2].plot(freqOut,np.abs(IVertFaceTot[:,2]),'r', label = "Simulated")
    axs[2].plot(freqOut,np.abs(IHorFaceTot[:,2]),'k', label = "Simulated")
    axs[2].plot(freqOut,np.ones((nFreq,))*8*np.pi/3*G*2800,'g',label = 'Theoretical')
    
    axs[2].set_xlim([0,8]);
    axs[2].set_ylabel('NN_z');
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_title('Z component');

def assembleNNAllRea(outReaPath,freqOut, nRea):
    """
    to be run at the end of all realizations
    compute the rms of surface and deep displacements
    
    """
    nFreq = len(freqOut)
    
    nnTotAllRea = np.zeros((nFreq,3))
    nnVolAllRea = np.zeros((nFreq,3))
    nnSurfAllRea = np.zeros((nFreq,3))

    for reaNo in range(0,nRea):
        sName = 'NNFullRea' + str(reaNo) + '.npz'
        data = np.load(os.path.join(outReaPath,sName));
        nnTot = np.abs(data["ITot"])
        nnVol = np.abs(data["IBlockTot"])
        nnSurf = np.abs(data["IVertFaceTot"]+data['IHorFaceTot'])
        nnTotAllRea = nnTotAllRea + nnTot**2
        nnVolAllRea = nnVolAllRea + nnVol**2
        nnSurfAllRea = nnSurfAllRea + nnSurf**2

    nnTotAllRea = np.sqrt(nnTotAllRea/nRea)
    nnVolAllRea = np.sqrt(nnVolAllRea/nRea)
    nnSurfAllRea = np.sqrt(nnSurfAllRea/nRea)

    freqOut = data["freqOut"]
    
    return nnTotAllRea, nnVolAllRea, nnSurfAllRea, freqOut  

def handleHorSurfNN(layer, layers, block, gridSize, nLayers, i, rhoAir, freqOut, cubeC, xSrc, ySrc, azSrc, srcMeta,
                    idxFreq, fMin, fMax, outDispPath, splitAll, xMaxGF, fInpPath, components, vR, nCPUDisp=4,
                    nCPUNN=4, useSimDisp=False, ifRay = 0, ifBody = 0, vP_for_test=4000, vS_for_test = 2000,
                    metaBody = None, chunk_size=20000, rCavity=20.0, computeStrategy="threading_shared",
                    saveHV=False):
    
    nFreq = len(freqOut)
    IHorFaceTot = np.zeros((nFreq,3),dtype=np.complex128)
    internalFlag = 0
    for f in block.horizontalFaces:
        if f.isBoundary:
            print('Horizontal face found')
            z_face = f.position
            
            # --- Determine which kind of face this is ---
            if np.isclose(z_face, layer.zTop):
                # Top of current layer
                if i == 0:
                    # Topmost layer → surface (air)
                    #print('I am in air')
                    rho_eff = rhoAir
                    internalFlag = 0
                else:
                    internalFlag = 1
                    continue  # skip internal top faces
                    
            elif np.isclose(z_face, layer.zBot):
                # Bottom of current layer
                if i < nLayers - 1:
                    #print('I am in midlayer situation, taking bottom surface of layer')
                    # Internal interface → difference term handled implicitly
                    rho_eff = layers[i + 1].rho - layer.rho
                    internalFlag = 1
                else:
                    #print('I am at the very bottom')
                    # Bottommost layer → outer boundary
                    rho_eff = layer.rho
                    internalFlag = 0
            else:
                continue  # shouldn't happen

            # --- Compute the horizontal surface NN for the outer bottom ot top face ---
            #print('rho_eff = '+str(rho_eff))
            if(not internalFlag):
                IHorFace = runHorSurfNNCompute(f, block.spaceType, gridSize, freqOut, cubeC, rho_eff, xSrc, ySrc,
                                           azSrc, srcMeta, idxFreq, fMin, fMax, outDispPath, splitAll,
                                           xMaxGF, fInpPath, components, vR, nCPUDisp=nCPUDisp, nCPUNN=nCPUNN,
                                           useSimDisp=useSimDisp, ifRay = ifRay, ifBody = ifBody, vP_for_test = vP_for_test,
                                           vS_for_test = vS_for_test, metaBody = metaBody, chunk_size=chunk_size,
                                           rCavity=rCavity,computeStrategy=computeStrategy, saveHV=saveHV)
            
                #
                IHorFaceTot = IHorFaceTot + IHorFace
    
    return IHorFaceTot


def runHorSurfNNCompute(face, spaceType, gridSize, freqOut, cubeC, rho, xSrc, ySrc, azSrc, srcMeta, idxFreq, 
                        fMin, fMax, outDispPath, splitAll, xMaxGF, fInpPath, components, vR, nCPUDisp=4, nCPUNN=4,
                        useSimDisp=False, ifRay = 0, ifBody = 0, vP_for_test = 4000, vS_for_test = 2000, metaBody = None,
                        chunk_size=20000, rCavity=20,computeStrategy="threading_shared",saveHV=False):
    """
    Compute NN from ONE horizontal interface (z = const), parallel over (x,y).

    Assumptions
    -----------
    - displacements for ALL (x,y) on that plane can be obtained in one go
      either from simDisp (recommended) or we synthesize a plane wave.
    """
    zFace = face.position # zFace is positive
    zFaceNeg = -zFace
    
    if(spaceType == 'lgwt'):
        #print('I am here!')
        totX = (face.xLim[1]-face.xLim[0])
        totY = (face.yLim[1]-face.yLim[0])
        
        nX = max(3,int(np.round(totX/gridSize)))
        nY = max(3,int(np.round(totY/gridSize)))
        #print('nX = '+str(nX) + 'nY = ' + str(nY))    
        gridX, gridXW = lgwtPoints(nX, face.xLim[0], face.xLim[1])
        gridY, gridYW = lgwtPoints(nY, face.yLim[0], face.yLim[1])
        #zlgwt = -np.abs(zlgwt)
    else:
        print('Uniform spacing')
        # the case of the block enclosing the cavity, no lgwt sampliing here
        # uniform sampling to be done here
        gridX, gridXW = make_centered_axis(face.xLim[0], face.xLim[1], 1.0, min_pts = 3)
        gridY, gridYW = make_centered_axis(face.yLim[0], face.yLim[1], 1.0, min_pts = 3)
    
    # build 2D grid
    X, Y = np.meshgrid(gridX, gridY, indexing="ij")
    WX, WY = np.meshgrid(gridXW, gridYW, indexing="ij")

    X_flat = X.ravel()
    Y_flat = Y.ravel()
    dS = (WX * WY).ravel()    # area weights
    # use -zFace
    zArr = np.full_like(X_flat, zFaceNeg, dtype=float)

    # cavity depth must be made negative for coordinate convention
    zCav = -(cubeC)
    
    # remove cavity footprint if this horizontal plane cuts it
    if spaceType == 'uniform' and rCavity is not None:
        rDist = np.sqrt(X_flat**2 + Y_flat**2 + (zFaceNeg - zCav)**2)
        mask = rDist >= rCavity
        X_flat = X_flat[mask]
        Y_flat = Y_flat[mask]
        zArr   = zArr[mask]
        dS     = dS[mask]

    # final coords
    rVec = np.column_stack((X_flat, Y_flat, zArr))   # (nPts, 3)
    nPts = rVec.shape[0]
    nFreq = len(freqOut)

    # get displacements
    if useSimDisp:
        # expected: (nFreq, nPts, 3)
        # note that zArr negative here, careful
        zUse = -zFaceNeg
        dispData, outDirUse, fUse = simDisp.computeFullDispN(zUse, rVec[:,0], rVec[:,1], xSrc, ySrc, azSrc, 
                    srcMeta, idxFreq, freqOut, fMin, fMax, outDispPath,
                    splitAll, xMaxGF, fInpPath, components, nCPU=nCPUDisp, nChunk=chunk_size,
                    minVel=100.0, computeStrategy=computeStrategy,saveHV=saveHV)
    else:
        if(ifRay):
            # pure Rayleigh wavefield for testing
            # note zFaceNeg is negative, so pass that
            phiSrc = srcMeta["phiSrc"]; ampSrc = srcMeta["ampSrc"]
            nSrc = len(azSrc)
            dispData = simDisp.computeRayleighDisp(zFaceNeg, rVec[:,0], rVec[:,1], nSrc, freqOut, azSrc,
                                                phiSrc, ampSrc, vR, vP_for_test, vS_for_test) 
        elif(ifBody):
            nSrc = len(azSrc)
            dispData = simDisp.computeFullBodyDisp(zFaceNeg, rVec[:,0], rVec[:,1], nSrc, freqOut, vP_for_test,
                                                   vS_for_test, metaBody)

    n_hat = face.normVector
    #print(n_hat)
    # run parallel surface integration
    #print("I am computing surface parallel")
    I_total = compute_surface_parallel(dispData, rVec, zCav, dS, n_hat, n_jobs=nCPUNN, chunk_size=chunk_size)

    I_total = I_total*rho
    
    return I_total
    
def compute_surface_parallel(dispData, rVec, zCav, dS, n_hat, n_jobs=4, chunk_size=20000):
    """
    Parallel surface NN over a horizontal face.

    Parameters
    ----------
    dispData : (nFreq, nPts, 3)
    rVec     : (nPts, 3)
    dS       : (nPts,)
    n_hat    : (3,)
    """
    nFreq, nPts, _ = dispData.shape
    I_total = np.zeros((nFreq, 3), dtype=np.complex128)

    def process_chunk(start, end):
        u_chunk = dispData[:, start:end, :]
        r_chunk = rVec[start:end, :]
        dS_chunk = dS[start:end]
        return computeVertSurfNN(u_chunk, r_chunk, zCav, n_hat, dS_chunk)

    starts = list(range(0, nPts, chunk_size))

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_chunk)(s, min(s + chunk_size, nPts))
        for s in starts
    )

    for i, res in enumerate(results):
        #print(f"[Main] Aggregating result from chunk {i+1}/{len(results)}")
        I_total += res
    
    return I_total
    
def runVertSurfNNCompute(face, gridSize, freqOut, cubeC, xSrc, ySrc, azSrc, srcMeta, idxFreq, fMin, fMax,
                         outDispPath, splitAll, xMaxGF, fInpPath, components, vR, nCPUDisp=4, nChunk=20000,
                         useSimDisp=False, ifRay = 0, ifBody = 0, vP_for_test=None, vS_for_test = None,
                         metaBody=None, computeStrategy="threading_shared",saveHV=False):
    """
    Surface NN for ONE vertical face, looping over depth.
    face.axis == 'x'  -> fixed x, vary y,z
    face.axis == 'y'  -> fixed y, vary x,z
    """
    nFreq = len(freqOut)
    I_total = np.zeros((nFreq, 3), dtype=np.complex128)

    if(face.axis=='x'):
        totY = face.yLim[1]-face.yLim[0]
        totZ = face.zLim[1]-face.zLim[0]
                    
        # at least three points in depth for integration
        nY = max(3,int(np.round(totY/gridSize)))
        nZ = max(3,int(np.round(totZ/gridSize)))
            
        faceCoords, faceWeightlgwt = lgwtPoints(nY, face.yLim[0], face.yLim[1])
        zlgwt, zWeightlgwt = lgwtPoints(nZ, face.zLim[0], face.zLim[1])
                
    if(face.axis=='y'):
        totX = face.xLim[1]-face.xLim[0]
        totZ = face.zLim[1]-face.zLim[0]
                    
        # at least three points in depth for integration
        nX = max(3,int(np.round(totX/gridSize)))
        nZ = max(3,int(np.round(totZ/gridSize)))
            
        faceCoords, faceWeightlgwt = lgwtPoints(nX, face.xLim[0], face.xLim[1])
        zlgwt, zWeightlgwt = lgwtPoints(nZ, face.zLim[0], face.zLim[1])

    # precompute dS along the face direction (x or y)
    # this is 1D: area element along-face
    dS_line = faceWeightlgwt  # shape (nPts,)

    # switch sign here, because NN geometry returns depth as positive
    zlgwt = -np.abs(zlgwt)
    zCav = -(cubeC)  # ensure cavity depth is negative
    
    for iz, (zNow, wz) in enumerate(zip(zlgwt, zWeightlgwt)):
        # build coords for THIS depth only
        if face.axis == "x":
            # x fixed, y varies
            xArr = np.full_like(faceCoords, face.position, dtype=float)
            yArr = faceCoords
            zArr = np.full_like(faceCoords, zNow, dtype=float)
        elif face.axis == "y":
            # y fixed, x varies
            xArr = faceCoords
            yArr = np.full_like(faceCoords, face.position, dtype=float)
            zArr = np.full_like(faceCoords, zNow, dtype=float)
        else:
            raise ValueError("face.axis must be 'x' or 'y' for vertical faces")

        rVec = np.column_stack((xArr, yArr, zArr))  # (nPts, 3)

        # full surface element for this depth slice
        # shape (nPts,)
        dS = dS_line * wz

        # get displacement at THIS depth
        if useSimDisp:
            # be carefull here zNow is negative
            zUse = -zNow
            disp_f, outDirUse, fUse = simDisp.computeFullDispN(zUse, rVec[:,0], rVec[:,1], xSrc, ySrc, azSrc, 
                    srcMeta, idxFreq, freqOut, fMin, fMax, outDispPath,
                    splitAll, xMaxGF, fInpPath, components, nCPU=nCPUDisp, nChunk=nChunk,
                    minVel=100.0, computeStrategy=computeStrategy, saveHV=saveHV)
        else:
            # plane wave test
            if(ifRay):
                # pure Rayleigh wavefield for testing
                # note zNow is negative, so pass that
                phiSrc = srcMeta["phiSrc"]; ampSrc = srcMeta["ampSrc"]
                nSrc = len(azSrc)
                disp_f = simDisp.computeRayleighDisp(zNow, rVec[:,0], rVec[:,1], nSrc, freqOut, azSrc,
                                                phiSrc, ampSrc, vR, vP_for_test, vS_for_test)
            elif(ifBody):
                nSrc = len(azSrc)
                disp_f = simDisp.computeFullBodyDisp(zNow, rVec[:,0], rVec[:,1], nSrc, freqOut, vP_for_test,
                                                     vS_for_test, metaBody)

        
        # now do the surface kernel for ALL freqs
        I_slice = computeVertSurfNN(disp_fxn=disp_f, rVec=rVec, zCav=zCav, n_hat=face.normVector, dS=dS)
        # add with density
        I_total += face.rhoOut * I_slice

    return I_total

def computeVertSurfNN(disp_fxn, rVec, zCav, n_hat, dS):
    """
    disp_fxn : (nFreq, nPts, 3)  complex
        displacement on the surface
    rVec : (nPts, 3)  float
        coordinates of surface points (in *global* coords)
    zCav : float
        cavity depth, BUT we subtract it here: r -> (r - r0)
        so pass +cubeC and we do r_z - cubeC
    n_hat : (3,) float
        outward normal of the surface
    dS : (nPts,) float
        surface area weights (per point)

    returns
    -------
    I_surf : (nFreq, 3) complex
    """
    nFreq, nPts, _ = disp_fxn.shape

    # shift to cavity
    rRel = rVec.copy()
    rRel[:, 2] = rRel[:, 2] - zCav  # z - z_cav

    # |r| and r/|r|^3
    rnorm = np.linalg.norm(rRel, axis=1)          # (nPts,)
    rnorm3 = rnorm**3                             # (nPts,)
    r_over_r3 = (rRel / rnorm3[:, None])          # (nPts, 3)

    # (u · n) for each freq, point
    # disp_fxn: (nFreq, nPts, 3)
    u_dot_n = np.einsum("fpk,k->fp", disp_fxn, n_hat)      # (nFreq, nPts)

    # scalar part: (u·n)/r^3
    scalar = u_dot_n / rnorm3[None, :]                     # (nFreq, nPts)

    # now multiply by r-vector → (nFreq, nPts, 3)
    I_pt = scalar[:, :, None] * rRel[None, :, :]           # (nFreq, nPts, 3)

    # apply surface weight dS
    I_pt *= dS[None, :, None]                              # (nFreq, nPts, 3)

    # integrate over surface points
    I_surf = I_pt.sum(axis=1)                              # (nFreq, 3)

    return I_surf
    
def runVolNNComputeBlock(block, gridSize, freqOut, cubeC, rCavity, xSrc, ySrc, azSrc, srcMeta,
                         idxFreq, fMin, fMax, outDispPath, splitAll, xMaxGF, fInpPath, components, vR, 
                         useSimDisp=False, ifRay = 0, ifBody = 0, nCPUDisp=4, nCPUNN=4, nChunk=20000,
                         metaBody=None, computeStrategy = 'threading_shared', saveHV = False):
    """
    Perform NN computation for one block.
    
    Parameters
    ----------
    block : NNGeometry.Block
        Current block object (contains x/y/z bounds, rho, vP, etc.)
    gridX, gridY : 1D arrays
        Quadrature or uniform grid coordinates in X and Y.
    gridXW, gridYW : 1D arrays
        Corresponding quadrature weights for X and Y.
    zlgwt, zWeightlgwt : 1D arrays
        Depth coordinates and weights (positive down).
    freqOut : ndarray
        Frequencies at which displacement is evaluated.
    cubeC : float
        Depth of cavity center (positive, in meters).
    rCavity : float
        Cavity radius (in meters).
    useSimDisp : bool
        If True, read displacement from file. Otherwise, generate synthetic plane wave.
    simDispFile : str
        Path to displacement file (if useSimDisp=True).
    """

    nFreq = len(freqOut)
    I_total = np.zeros((nFreq, 3), dtype=np.complex128)

    if(block.spaceType == 'lgwt'):
        #print('I am here!')
        totX = (block.xMax-block.xMin)
        totY = (block.yMax-block.yMin)
        totZ = (block.zMax-block.zMin)
            
        nX = max(3,int(np.round(totX/gridSize)))
        nY = max(3,int(np.round(totY/gridSize)))
        nZ = max(3,int(np.round(totZ/gridSize))) # at least three points in depth for integration
            
        gridX, gridXW = lgwtPoints(nX, block.xMin, block.xMax)
        gridY, gridYW = lgwtPoints(nY, block.yMin, block.yMax)
        zlgwt, zWeightlgwt = lgwtPoints(nZ, block.zMin, block.zMax)
        #zlgwt = -np.abs(zlgwt)
    else:
        print('Uniform spacing')
        # the case of the block enclosing the cavity, no lgwt sampliing here
        # uniform sampling to be done here

        gridX, gridXW = make_centered_axis(block.xMin, block.xMax, 1.0, min_pts = 3)
        gridY, gridYW = make_centered_axis(block.yMin, block.yMax, 1.0, min_pts = 3)
        zlgwt, zWeightlgwt = make_centered_axis(block.zMin, block.zMax, 1.0, min_pts = 3)
    
    # --- Construct full horizontal grid ---
    gridXMat, gridYMat = np.meshgrid(gridX, gridY, indexing='ij')
    gridXWMat, gridYWMat = np.meshgrid(gridXW, gridYW, indexing='ij')
    gridX_flat = gridXMat.ravel()
    gridY_flat = gridYMat.ravel()
    gridW_flat = (gridXWMat * gridYWMat).ravel()

    # --- Flip z to negative (depths below surface) ---
    zlgwt = -np.abs(zlgwt)
    zCav = -(cubeC)  # ensure cavity depth is negative

    # --- Loop over depth slices ---
    for d in range(len(zlgwt)):
        zNow = zlgwt[d]
        #print("Depth = " + str(zNow));
        zWeight = zWeightlgwt[d]
        zWNow = zWeight * gridW_flat

        # full coordinate vectors
        rVec = np.column_stack((gridX_flat, gridY_flat, np.full_like(gridX_flat, zNow)))
        dV = zWNow  # base differential volume weights

        # ============================================================
        # CAVITY REMOVAL
        # ============================================================
        # Apply only to the block containing the cavity (uniform grid)
        if block.spaceType == 'uniform':
            # compute distance from cavity center
            rDist = np.sqrt(gridX_flat**2 + gridY_flat**2 + (zNow - zCav)**2)
            mask = rDist >= rCavity  # keep only points OUTSIDE cavity

            # apply mask to geometry and weights
            rVec = rVec[mask, :]
            dV = dV[mask]
            if len(rVec) == 0:
                print(f"[WARNING] All points removed at z={zNow:.2f} in cavity block — skipping this slice.")
                continue
        # ============================================================

        # --- Generate or read displacements ---
        if useSimDisp:
            # TODO: memory-mapped read from file (later)
            # be careful here zNow is negative
            zUse = -zNow
            #print('Started disp computation at depth ' + str(zUse))
            dispData, outDirUse, fUse = simDisp.computeFullDispN(zUse, rVec[:,0], rVec[:,1], xSrc, ySrc, azSrc, 
                    srcMeta, idxFreq, freqOut, fMin, fMax, outDispPath,
                    splitAll, xMaxGF, fInpPath, components, nCPU=nCPUDisp, nChunk=nChunk,
                    minVel=100.0, computeStrategy=computeStrategy,saveHV=saveHV)
            #print('Ended disp computation at depth ' + str(zUse))
        else:
            if(ifRay):
                # pure Rayleigh wavefield for testing
                # note zNow is negative, so pass that
                phiSrc = srcMeta["phiSrc"]; ampSrc = srcMeta["ampSrc"]
                nSrc = len(azSrc)
                dispData = simDisp.computeRayleighDisp(zNow, rVec[:,0], rVec[:,1], nSrc, freqOut, azSrc,
                                                phiSrc, ampSrc, vR, block.vP, block.vS)
            elif(ifBody):
                nSrc = len(azSrc)
                dispData = simDisp.computeFullBodyDisp(zNow, rVec[:,0], rVec[:,1], nSrc, freqOut,
                                                       block.vP, block.vS, metaBody)
            else:
                raise ValueError("ifRay and ifBody can't be set to 0 when useQseis=False")

        # --- Volume integral for this depth slice ---
        I_total += compute_volume_parallel(dispData, rVec, zCav, block.rho, dV, n_jobs=nCPUNN)
        #print("grid points = " + str(len(rVec)));
    
    return I_total
    
def compute_volume_parallel(dispData, rVec, zCav, rho, dV, n_jobs=4, chunk_size=20000):
    """
    Parallel computation of volume NN contribution using joblib.
    dispData: (nFreq, nGrid, nComp)
    rVec: (nGrid, 3)
    dV:   (nGrid,)
    """
    nFreq, nGrid, _ = dispData.shape
    I_total = np.zeros((nFreq, 3), dtype=np.complex128)

    def process_chunk(start, end):
        """Worker task for a subset of grid points."""
        pid = os.getpid()
        npts = end - start
        #print(f"[Worker {pid}] Processing chunk {start}:{end} ({npts} points)")

        u_chunk = dispData[:, start:end, :]
        r_chunk = rVec[start:end, :]
        dV_chunk = dV[start:end]

        result = getVolNN_chunk(u_chunk, r_chunk, zCav, rho, dV_chunk)

        #print(f"[Worker {pid}] Finished chunk {start}:{end}")
        return result

    # Build all chunk boundaries
    chunk_starts = list(range(0, nGrid, chunk_size))
    #print(f"Total {len(chunk_starts)} chunks for {nGrid} grid points, "
    #      f"running with {n_jobs} workers...")

    # Dispatch chunks in parallel
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(process_chunk)(start, min(start + chunk_size, nGrid))
        for start in chunk_starts
    )

    # Aggregate results
    for i, res in enumerate(results):
        #print(f"[Main] Aggregating result from chunk {i+1}/{len(results)}")
        I_total += res

    #print("[Main] All chunks complete. Total integrated volume field ready.")
    return I_total
    
def getVolNN_chunk(u_chunk, r_chunk, zCav, rho, dV):
    """
    Compute the volume NN contribution for a spatial chunk, across all frequencies.
    u_chunk : (nFreq, nChunk, 3)
    r_chunk : (nChunk, 3)
    zCav    : cavity depth (m)
    rho     : density (kg/m³)
    dV      : (nChunk,) volume weights for integration
    """
    rRel = r_chunk.copy()
    rRel[:, 2] -= zCav
    rDist = np.linalg.norm(rRel, axis=1)
    rCap = rRel / rDist[:, np.newaxis]
    rDist3 = rDist**3

    # broadcast along frequency
    rCap = rCap[np.newaxis, :, :]         # (1, nChunk, 3)
    rDist3 = rDist3[np.newaxis, :, np.newaxis]
    dV = dV[np.newaxis, :, np.newaxis]

    term1 = u_chunk / rDist3
    dot_ru = np.sum(rCap * u_chunk, axis=2, keepdims=True)
    term2 = 3.0 * rCap * dot_ru / rDist3

    I_chunk = term1-term2
    I_local = rho * (term1 - term2) * dV   # (nFreq, nChunk, 3)
    I_chunk = np.sum(I_local, axis=1)      # (nFreq, 3)
    return I_chunk

def getAnalyticRayNN(config):
    # analytic Rayleigh wave calculation
    freqs = np.arange(config.fMin,config.fMax,0.1);
    nF = len(freqs)
    thickness = config.thickness
    vP = config.vP
    vS = config.vS
    rho = config.rho

    periods = 1/freqs
    h = config.cavityDepth
    gamma = 0.8

    vR = surf96(thickness,vP,vS,rho,periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
    qP = (2*np.pi*freqs)/(vR*vP[0])*np.sqrt(vP[0]**2 - vR**2)
    qS = (2*np.pi*freqs)/(vR*vS[0])*np.sqrt(vS[0]**2 - vR**2)
    ki = np.sqrt(qP/qS)
    kR = (2*np.pi*freqs)/vR
    RNum = -kR*(1+ki)*np.exp(-kR*h) + 2/3*(2*kR*np.exp(-qP*h) + ki*qS*np.exp(-qS*h))
    RDen = kR*(1-ki)
    R = np.abs(RNum/RDen)
    NNR = np.sqrt(2)*np.pi*const.G*gamma*rho[0]*R
    return NNR,freqs

def getAnalyticBodyNN(config,p=1/3):
    # analytic body wave NN calculation
    freqs = np.arange(config.fMin,config.fMax,0.1)
    nF = len(freqs)
    minNN =(4*np.pi/3*const.G*config.rho[0])*np.ones((nF,))
    bodyNN = (4*np.pi/3*const.G*config.rho[0]*np.sqrt(3*p+1))*np.ones((nF,)) 
    return bodyNN, minNN, freqs

def make_centered_axis(xmin, xmax, target_dx=1.0, min_pts=3):
    """
    Uniform, cell-centered grid inside [xmin, xmax].
    Returns centers and equal weights (~target_dx).
    """
    L = xmax - xmin
    n_int = max(min_pts, int(np.round(L / target_dx)))
    dx = L / n_int
    # center points shifted half a cell inside each end
    x = xmin + dx / 2 + dx * np.arange(n_int)
    w = np.full_like(x, dx)
    return x, w

    
def lgwtPoints(N, a, b):
    """
    generate points within an interval based on Gauss-Legendre quadrature rule
    # N is the number of points in the interval (a,b)
    
    """
    N = N-1;
    N1 = N+1;
    N2 = N+2;

    xu = np.linspace(-1,1,N1);

    # Initial guess
    y = np.cos((2*np.arange(N1) + 1)*np.pi/(2*N + 2)) + \
        0.27 / N1*np.sin(np.pi*xu*N/N2);

    y0 = 2.0;
    L = np.zeros((N1,N2));
    Lp = np.zeros(N1);

    # Iterate with Newton-Raphson until convergence
    while np.max(np.abs(y-y0)) > np.finfo(float).eps:
        y0 = y.copy();

        L[:,0] = 1.0;
        L[:,1] = y;

        for k in range(1, N1):
            L[:,k+1] = ((2*k+1)*y*L[:,k] - k*L[:,k-1])/(k+1);

        Lp = N2*(L[:,N1-1]-y*L[:,N1]) / (1-y**2);

        y = y0 - L[:,N1]/Lp

    # Map from [-1, 1] to [a, b]
    x = (a*(1-y) + b*(1+y)) / 2

    # Compute the weights
    w = (b-a)/((1-y**2)*Lp**2)*(N2/N1)**2

    # Sort x and reorder weights
    idx = np.argsort(x)
    x = x[idx]
    w = w[idx]

    return x, w;

def getGridSize(thickness,vP, vS, rho, fMin, fMax, lambdaFrac, lambdaRes):
    """
    the function returns the grid size per layer by comparing the S-wave velocities and
    the frequency dependent Rayleigh wave velocities
    thickness is the thickness of each layer in meters,set last layer thickness to 0 for halfspace
    vP, vS are the elastic P-wave and S-wave velocities of each layer in m/s, including half space
    rho is the density of each layer in kg/m^3
    fMin, fMax are the minimum ana maximum frequency of the NN band
    lambdaFrac is between 0 and 1, denoting the sensitivity of Rayleigh waves in depth
    if lambdaFrac=1/3, then sensitivity of Rayliegh wave is upto that depth at that frequency
    lambdaRes is the number of points per lambda to be resolved using the grid
    typical values of lambdaRes is between 4 and 6, increasing it will only increase the computational
    and the memory load
    
    """
    # main code
    [freqMin, vRMin] = getMinVelFreq(thickness,vP, vS, rho, fMin, fMax, lambdaFrac);
    lambdaRMin = vRMin/freqMin*1000; # in meters
    
    lambdaSMin = vS/fMax; # in meters
    
    lambdaMin = np.minimum(lambdaRMin,lambdaSMin);

    gridSize = lambdaMin/lambdaRes;
    
    return gridSize;

def getMinVelFreq(thickness,vP, vS, rho, fMin, fMax, lambdaFrac):
    """
    # thickness is the thickness of each layer in meters,set last layer thickness to 0 for halfspace
    # vP, vS are the elastic P-wave and S-wave velocities of each layer in m/s, including half space
    # rho is the density of each layer in kg/m^3
    # fMin, fMax are the minimum ana maximum frequency of the NN band
    # lambdaFrac is between 0 and 1, denoting the sensitivity of Rayleigh waves in depth
    # if lambdaFrac=1/3, then sensitivity of Rayliegh wave is upto that depth at that frequency
    """
    # main code
    # convert everything to units suitable for surf96, CPS.330 code
    thick = thickness/1000; # converted to kilometers
    vP = vP/1000; vS = vS/1000; # converted to km/s
    rho = rho/1000; # converted to gm/cc
    
    # Periods we are interested in
    freqs = np.arange(fMin,fMax,0.2);
    
    periods = 1/freqs;

    vDispRay = surf96(thick,vP,vS,rho,periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
    #print(vDispRay)
    # surf96 can fail if it is Love wave for homogeneous half-space for
    # put a check
    homoFlag = 1
    for i in range(0,(len(vS)-1)):
        if((vS[i+1]-vS[i])!=0):
            homoFlag = 0
    if not homoFlag:
        vDispLove = surf96(thick,vP,vS,rho,periods,wave="love",mode=1,velocity="phase",flat_earth=True)
    else:
        vDispLove = vDispRay
    #print(vDispLove)
    
    # get the minimum of the Rayleigh and Love wave phase velocity
    vMinRayLove = np.minimum(vDispRay, vDispLove);
    
    #print(vMinRayLove)
    
    lambdaVal = vMinRayLove/freqs;
    lambdaValBy3 = lambdaVal*lambdaFrac;
    depths = np.cumsum(thick);
    depths = np.insert(depths,0,0.0,axis=0);
    vMin = np.zeros((len(depths)-1),);
    freqMin = np.zeros((len(depths)-1),);
    
    for depthNo in range(0,(len(depths)-1)):
        fInd = np.where((lambdaValBy3<=depths[depthNo+1]) & (lambdaValBy3>=depths[depthNo]))[0]
        if(len(fInd)>0):
            vMin[depthNo] = vMinRayLove[fInd[-1]]
            freqMin[depthNo] = freqs[fInd[-1]];
            if(vMin[depthNo]>vS[depthNo]):
                vMin[depthNo] = vS[depthNo];
        else:
            vMin[depthNo] = vS[depthNo];
            freqMin[depthNo] = fMax;
    return freqMin, vMin;    
if __name__ == "__main__":
    main()