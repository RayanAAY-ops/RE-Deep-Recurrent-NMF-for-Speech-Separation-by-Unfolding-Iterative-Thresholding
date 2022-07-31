
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.signal import butter, lfilter, freqz
from sklearn.utils.extmath import randomized_svd,squared_norm
import math
import numpy as np
from numpy import linalg as LA

def Viz_Y(t,f,Y, vmin=0, vmax=20):
    plt.figure(figsize=(20,7))
    plt.pcolormesh(t, f, Y,vmin=0, vmax=20, shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    

def Reconstruct(B,G,Ns,Nm,Yabs,p):
    
    numerators=[]
    B1=B[:,:Ns]
    B2=B[:,Ns:]
    G1=G[:Ns,:]
    G2=G[Ns:,:]
    
    
    numerators.append(np.power(np.matmul(B1,G1),p))
    numerators.append(np.power(np.matmul(B2,G2),p))

    denominator = np.power(np.matmul(B1,G1),p)+np.power(np.matmul(B2,G2),p)
  
    

    Sources=[]
    Masks=[]
    for i in range(2):

        Sources.append(np.multiply(numerators[i]/denominator,Yabs))
        Masks.append(numerators[i]/denominator)

    #print('Source shape = {}'.format(Sources[0].shape))
    
    return Sources,Masks


def Reconstruct_2comp(n_components,B,G, Yabs):
    
    percents=[]
    numerators=[]
    
    denominator = np.zeros((B.shape[0],G.shape[1]))
    for i in range(n_components):
    
        denominator += np.matmul(B[:,i].reshape((B.shape[0],1)),G[i,:].reshape((1,G.shape[1])))
        numerator = np.matmul(B[:,i].reshape((B.shape[0],1)),G[i,:].reshape((1,G.shape[1])))
        numerators.append(numerator)

        
    for i in range(n_components):
        
        percents.append(numerators[i]/(denominator+0.001))
    
    
    Sources=[]

    for i in range(n_components):

        Sources.append(np.multiply(percents[i],Yabs))

    print('Source shape = {}'.format(Sources[0].shape))
    
    return Sources


def SMR(speech, music):
    
    """
    Function that takes music and speech signals.
    returns SMR in db
    """
    speech_power = LA.norm(speech,2)
    music_power = LA.norm(music,2)
    SMR_db=10*np.log10(speech_power/music_power)
    print('SMR = {:.2f}'.format(SMR_db))
    
    return SMR_db

def SDR(s_est, s):
    """
    Function that takes original and estimated spectrogram
    returns SDR in DB
    """
    
    signal_power = LA.norm(s,2)
    distorsion_power = LA.norm(s_est - s,2) 
    SDR_db=10*np.log10(signal_power/distorsion_power)
    
    return SDR_db

def plot_SDR(list_smr,list_music,list_speech):
  plt.figure(figsize=(10,5))
  sns.set_style("darkgrid")

  ax1 = sns.lineplot(list_smr,list_music)
  ax2 = sns.lineplot(list_smr,list_speech)
  ax1.set(xlabel='SMR  (db)', ylabel='SMR  (db)')
  ax1.legend(["Estimated MUSIC signal","Estimated SPEECH signal"])
  plt.show()
  
def get_mixed_signal(speech, music, SMR_db):
    """
    Function taht takes the speech and music signal alongside the SMR_db
    returns the mixed signal and the scaled speech
    """
    smr = 10**(SMR_db/10)
    speech_power = LA.norm(speech,2)
    music_power = LA.norm(music,2)
    scale = smr * music_power / speech_power
    

    
    if SMR_db < 0 :
        mixed = scale* speech + music
        speech_scaled=scale*speech
        SMR(speech_scaled,music)
        return mixed,speech_scaled,music
    
    if SMR_db >= 0 :
        
        mixed =  speech + music * (1/scale)
        music_scaled=(1/scale) * music
        SMR(speech,music_scaled)
        return mixed,speech,music_scaled

def ReconstructSoft(B,G,Ns,Nm,Yabs,p):
    
    numerators=[]
    B1=B[:,:Ns]
    B2=B[:,Ns:]
    G1=G[:Ns,:]
    G2=G[Ns:,:]
    
    
    numerators.append(np.exp(np.power(np.matmul(B1,G1),p)))
    numerators.append(np.exp(np.power(np.matmul(B2,G2),p)))

    denominator = np.power(np.exp(np.matmul(B1,G1)),p)+np.power(np.exp(np.matmul(B2,G2)),p)
  
    

    Sources=[]
    Masks=[]
    for i in range(2):

        Sources.append(np.multiply(numerators[i]/denominator,Yabs))
        Masks.append(numerators[i]/denominator)

    #print('Source shape = {}'.format(Sources[0].shape))
    
    return Sources,Masks

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def norm(x):
  return math.sqrt(squared_norm(x))
def nndsvd(X,n_components):  
    eps = 1e-7 
    U, S, V = randomized_svd(X, n_components, random_state=7)
    W = np.zeros_like(U)
    H = np.zeros_like(V)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0
    return W,H
