import numpy as np
from matplotlib import pyplot as plt


def transformeesDeFourierRangeDoppler(signalRecu, zeroPadding):
    """
    Méthode effectuant les différentes transformées de Fourier et multiplication par la fenêtre de Hamming nécessaires
    pour créer le diagramme Range-Doppler. Si nécessaire, c’est cette fonction qui gère le zéro-padding.

    :param signalRecu: signal sur lequel les calculs vont êtres effectués
    :param zeroPadding: True si on désire effectuer du zéro-padding sur le slow time et False sinon
    :return: matrice contenant le diagramme Range-Doppler
    """
    L, N = signalRecu.shape

    if zeroPadding:
        signalRecu = np.hstack((signalRecu, np.zeros((L, N * 7))))
        N *= 8

    # range win processing
    signalRecu = np.multiply(signalRecu.T, np.kaiser(L, 8.6)).T

    # range fft processing
    signalRecu = np.fft.fft(signalRecu, L, axis=0)

    # doppler win processing
    # signalRecu = np.multiply(signalRecu, np.kaiser(N, 8.6))

    # doppler fft processing
    signalRecu = np.fft.fft(signalRecu, N, axis=1)

    signalRecu = np.flip(np.roll(signalRecu, shift=N // 2, axis=1), axis=0)
    signalRecu = np.abs(signalRecu)

    return signalRecu, L, N


def imshow(image, xlabel=None, ylabel=None, title=None, extent=None, animation=True, vmin=None, vmax=None):
    """
    Méthode affichant une image.

    :param image: image a afficher
    :param xlabel: nom de l'axe x
    :param ylabel: nom de l'axe y
    :param title: titre du graphique
    :param extent: valeurs limites sur les axes x et y
    :param animation: True si l'image à afficher fait partie d'une animation et False sinon
    :param vmin: amplitude minimum de l'image
    :param vmax: amplitude maximum de l'image
    """
    plt.clf()
    plt.imshow(image, extent=extent, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    if animation:
        plt.pause(0.05)
    else:
        plt.show()


def plot(x, yList, xlabel=None, ylabel=None, title=None, labelList=None, zoom=False, ylim=None, animation=True):
    """
    Méthode affichant une ou plusieurs courbes sur un graphique.

    :param x: liste contenant les valeurs de l'axe x
    :param yList: liste de liste contenant les différentes courbes à afficher
    :param xlabel: nom de l'axe x
    :param ylabel: nom de l'axe y
    :param title: titre du graphique
    :param labelList: nom des courbes affichées
    :param zoom: True si l'utilisateur veut que les graphiques soient zoomés et False sinon
    :param ylim: bornes de l'axe y
    :param animation: True si les graphiques à afficher font parties d'une animation et False sinon
    """
    plt.clf()
    M = x.size//256
    for i, y in enumerate(yList):
        if zoom:
            plt.plot(x[128 * M - 12 * M:128 * M + 12 * M], y[128 * M - 12 * M:128 * M + 12 * M], label=labelList[i])
        else:
            plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.legend()
    plt.title(title)
    if animation:
        plt.pause(0.05)
    else:
        plt.show()
