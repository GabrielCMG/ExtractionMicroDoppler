import struct
import numpy as np
import matplotlib.pyplot as plt

from ExtractionData import ExtractionData
from Utilitaire import transformeesDeFourierRangeDoppler, imshow, plot


class TraitementData:
    """
    Classe contenant plusieurs méthodes permettant de traiter les données issues de mesures radar. Elle est adaptée au
    modèle de radar KMD2, mais peut être adaptée à d'autres modèles relativement facilement.
    """

    def __init__(self, nomFichier):
        self.data = ExtractionData(nomFichier)
        self.signalRecuMean = None
        self.signalRecuTriple = None
        self.dR = None
        self.Rmax = None
        self.dV = None
        self.Vmax = None
        self.extractionDataADC()
        self.trouverResolution()
        self.T, self.Dv = self.determinerSinc()
        self.targetListGlobal = []

    def extractionDataADC(self):
        """
        Méthode transformant les données du radar et les stockant dans deux listes. Une première contenant la moyenne du
        signal sur les trois antennes et une seconde contenant les signaux issus des trois antennes séparément.
        """
        self.signalRecuMean = []
        self.signalRecuTriple = []

        for rawADC in self.data.RADC:
            rawADC = np.frombuffer(rawADC, dtype=np.uint16)

            # Extraction du canal I pour les trois récepteurs
            rawADC_I_channel = rawADC[::2]
            rx1_I = rawADC_I_channel[:256 * 256]
            rx2_I = rawADC_I_channel[256 * 256:256 * 256 * 2]
            rx3_I = rawADC_I_channel[256 * 256 * 2:256 * 256 * 3]

            # Extraction du canal Q pour les trois récepteurs
            rawADC_Q_channel = rawADC[1::2]
            rx1_Q = rawADC_Q_channel[:256 * 256]
            rx2_Q = rawADC_Q_channel[256 * 256:256 * 256 * 2]
            rx3_Q = rawADC_Q_channel[256 * 256 * 2:256 * 256 * 3]

            # Reconstitution du signal reçu
            signalRecuRaw = (rx1_I + rx2_I + rx3_I + 1j * (rx1_Q + rx2_Q + rx3_Q)) / 3

            # Mise sous une forme plus facile à utiliser du signal reçu
            signalRecu = signalRecuRaw.reshape(256, 256).T

            self.signalRecuMean.append(signalRecu)

            self.signalRecuTriple.append([(rx1_I + 1j * rx1_Q).reshape(256, 256).T,
                                          (rx2_I + 1j * rx2_Q).reshape(256, 256).T,
                                          (rx3_I + 1j * rx3_Q).reshape(256, 256).T])

    def afficherRawData(self):
        """
        Méthode affichant les données brutes issues du radar.
        """
        for rawADC in self.signalRecuMean:
            plt.clf()
            plt.imshow(np.abs(rawADC))
            plt.xlabel('Slow Time')
            plt.ylabel('Fast Time')
            plt.pause(0.05)

    def afficherChirp(self, start, number):
        """
        Méthode affichant une animation de l'évolution d'une sélection de chirp dans les frames au cours du temps.

        :param int start: indice du premier chirp à afficher
        :param int number: nombre de chirps successifs à afficher
        """
        for rawADC in self.signalRecuMean:
            chirp = rawADC[:, start:start + number].T.reshape(-1, 1)

            plt.clf()
            plt.plot(np.real(chirp), label="Partie réelle")
            plt.plot(np.imag(chirp), label="Partie imaginaire")
            plt.xlabel('Échantillons')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.title("Représentation des parties réelles et imaginaires d'un chirp")
            plt.show()

    def afficherRangeDopplerRadar(self, zeroDoppler=True):
        """
        Méthode affichant le diagramme Range-Doppler donné par le radar. Aucun traitement supplémentaire n'est effectué
        sur l'information étant donné que tout le traitement est effectué dans le radar. La méthode propose en
        complément de cacher la bande de fréquences correspondant au zéro Doppler.

        :param bool zeroDoppler: Affichage de la bande correspondant au zero Doppler ou non
        """
        for dataMRD in self.data.RMRD:
            dataMRD = np.frombuffer(dataMRD, dtype=np.uint32).reshape(256, 256)

            if not zeroDoppler:
                for idx, line in enumerate(dataMRD):
                    dataMRD[idx] = self.supprimerZeroDoppler(line, 256)

            dataMRD = 10 * np.log10(np.abs(dataMRD) + 1)

            imshow(dataMRD, extent=[-self.Vmax, self.Vmax, 0, self.Rmax],
                   vmin=np.quantile(dataMRD, 0.3), vmax=np.quantile(dataMRD, 0.9),
                   xlabel='Speed [m/s]', ylabel='Range [m]', title="Diagramme Range-Doppler du radar")

    def afficherRangeDoppler(self, zeroPadding=True, zeroDoppler=True, methodeChirpMoyen=True):
        """
        Méthode calculant le diagramme Range-Doppler provenant du signal moyen des trois antennes et l'affichant. Le
        diagramme est animé. On peut choisir de supprimer le zéro-doppler et d'effectuer du zéro-padding.

        :param bool zeroPadding: Utilisation de zéro-padding sur le slow time ou non
        :param bool zeroDoppler: Affichage de la bande correspondant au zéro Doppler ou non
        :param methodeChirpMoyen: Utilisation de la méthode de soustraction du chirp moyen si vrai et de la soustraction du sinus cardinal sinon
        """
        for i, signalRecu in enumerate(self.signalRecuMean):

            if not zeroDoppler and methodeChirpMoyen:
                signalRecu -= self.chirpMoyen(i)

            signalRecu, L, N = transformeesDeFourierRangeDoppler(signalRecu, zeroPadding)
            signalRecu = np.abs(signalRecu)

            if not zeroDoppler and not methodeChirpMoyen:
                for idx, line in enumerate(signalRecu):
                    signalRecu[idx] = self.supprimerZeroDoppler(line, N)

            signalRecu = 10 * np.log10(np.abs(signalRecu) + 1)

            imshow(signalRecu, xlabel='Speed [m/s]', ylabel='Range [m]',
                   title="Diagramme Range-Doppler", animation=True)

    def afficherRangeDopplerAntenne(self, antennes, zeroPadding=True, zeroDoppler=True):
        """
        Méthode calculant la moyenne des trois diagrammes Range-Doppler provenant des signaux des trois antennes et
        l'affichant. Le diagramme est animé. On peut choisir de supprimer le zéro-doppler et d'effectuer du
        zéro-padding. On peut également choisir d'ignorer certaines antennes.

        :param list[int] antennes: liste des antennes à prendre en compte
        :param bool zeroPadding: True si on désire effectuer du zéro-padding sur le slow time et False sinon
        :param bool zeroDoppler: True si on désire supprimer le zéro-doppler et False sinon
        """
        for i, signalRecu in enumerate(self.signalRecuTriple):
            if zeroPadding:
                signalTotal = np.zeros((256, 1024), dtype='complex128')
            else:
                signalTotal = np.zeros((256, 256), dtype='complex128')

            for i in antennes:
                signal = signalRecu[i - 1]

                signal, L, N = transformeesDeFourierRangeDoppler(signal, zeroPadding)

                signalTotal += signal

            signalTotal /= len(signalRecu)

            if not zeroDoppler:
                for idx, line in enumerate(signalRecu):
                    signalRecu[idx] = self.supprimerZeroDoppler(line, signalTotal.shape[1])

            signalRecu = 10 * np.log10(np.abs(signalTotal) + 1)

            imshow(signalRecu, extent=[-self.Vmax, self.Vmax, 0, self.Rmax],
                   vmin=np.quantile(signalRecu, 0.3), vmax=np.quantile(signalRecu, 0.9),
                   xlabel='Speed [m/s]', ylabel='Range [m]', title="Diagramme Range-Doppler (moyenne par antenne)")

    def afficherDopplerTime(self, dist=None, zeroPadding=True, zeroDoppler=True, methodeChirpMoyen=True):
        """
        Méthode affichant le diagramme Doppler-Time d'une mesure, soit à une distance donnée, soit à une distance
        déterminée automatiquement. On peut choisir de supprimer le zéro-doppler et d'effectuer du zéro-padding.

        La détermination automatique de la distance à laquelle se trouve le drone est pour l'instant fonctionnelle
        uniquement si la cible (le drone dans mon cas) est à une distance fixe du radar. Cela pourrait être amélioré
        utilisant la fonctionnalité de tracking des cibles disponibles directement depuis le radar.

        :param int dist: distance à laquelle se trouve la cible (None si calculée automatiquement)
        :param bool zeroPadding: True si on désire effectuer du zéro-padding sur le slow time et False sinon
        :param bool zeroDoppler: True si on désire supprimer le zéro-doppler et False sinon
        :param methodeChirpMoyen: Utilisation de la méthode de soustraction du chirp moyen si vrai et de la soustraction du sinus cardinal sinon
        """
        diagrammeDopplerTime = None
        t = 0
        dictRange = dict()

        for i, signalRecu in enumerate(self.signalRecuMean):

            if not zeroDoppler and methodeChirpMoyen:
                signalRecu -= self.chirpMoyen(i)

            if t >= 5:
                break
            t += 0.05

            signalRecu, L, N = transformeesDeFourierRangeDoppler(signalRecu, zeroPadding)
            signalRecu = np.abs(signalRecu)

            if dist is None:
                indiceRange = np.argmax(np.quantile(signalRecu[5:250, 100:150], 0.50, axis=1)) + 5
                if indiceRange in dictRange.keys():
                    dictRange[indiceRange] += 1
                else:
                    dictRange[indiceRange] = 1
                indiceRange = max(dictRange, key=dictRange.get)
            else:
                indiceRange = dist

            signalRange = signalRecu[indiceRange]

            if not zeroDoppler and not methodeChirpMoyen:
                signalRange = self.supprimerZeroDoppler(signalRange, N)

            signalRange = 10 * np.log(np.abs(signalRange) + 1)

            if diagrammeDopplerTime is None:
                diagrammeDopplerTime = signalRange.reshape(-1, 1)
            else:
                diagrammeDopplerTime = np.hstack((diagrammeDopplerTime, signalRange.reshape(-1, 1)))

        Tmax = t

        imshow(diagrammeDopplerTime, extent=[0., Tmax, -self.Vmax, self.Vmax], animation=False,
               vmin=40, xlabel='Time [sec]', ylabel='Speed [m/s]', title="Diagramme Doppler-Time")

    def afficherSpectrogrammeSpeed(self, dist=None, zeroPadding=True, zeroDoppler=True, methodeChirpMoyen=True,
                                   zoom=False, sincCourbe=True):
        """
        Méthode affichant le diagramme Doppler-Time d'une mesure, soit à une distance donnée, soit à une distance
        déterminée automatiquement.

        :param int dist: distance à laquelle se trouve la cible (None si calculée automatiquement)
        :param bool zeroPadding: True si on désire effectuer du zéro-padding sur le slow time et False sinon
        :param bool zeroDoppler: True si on désire supprimer le zéro-doppler et False sinon
        :param methodeChirpMoyen: Utilisation de la méthode de soustraction du chirp moyen si vrai et de la soustraction du sinus cardinal sinon
        :param bool zoom: True si on désire effectuer un zoom autour du zéro-doppler et False sinon
        :param bool sincCourbe: True si on désire afficher le sinus cardinal approchant la signature de la cible et False sinon
        """
        sommeSpectrogrammes = None

        t = 0
        N = 0
        dictRange = dict()

        for i, signalRecu in enumerate(self.signalRecuMean):

            if not zeroDoppler and methodeChirpMoyen:
                signalRecu -= self.chirpMoyen(i)

            if t >= 5:
                break
            t += 0.05

            signalRecu, L, N = transformeesDeFourierRangeDoppler(signalRecu, zeroPadding)
            signalRecu = np.abs(signalRecu)

            if dist is None:
                indiceRange = np.argmax(np.max(signalRecu[5:250, 100:150], axis=1)) + 5
                if indiceRange in dictRange.keys():
                    dictRange[indiceRange] += 1
                else:
                    dictRange[indiceRange] = 1
                indiceRange = dictRange[max(dictRange, key=dictRange.get)]
            else:
                indiceRange = dist

            signalRange = signalRecu[indiceRange]

            vitesse = np.linspace(-self.Vmax, self.Vmax, signalRange.size)

            if not zeroDoppler and not methodeChirpMoyen:
                self.supprimerZeroDoppler(signalRange, N)

            # Calcul de la somme de tous les spectrogrammes pour obtenir le spectre moyen
            if sommeSpectrogrammes is None:
                sommeSpectrogrammes = signalRange
            else:
                sommeSpectrogrammes += signalRange

            # Affichage du signal à chaque frame
            if sincCourbe:
                sinc = self.calculerSinc(vitesse, signalRange, N)

                courbes, labels = [signalRange, sinc], ["Spectre", "Sinus cardinal"]
            else:
                courbes, labels = [signalRange], ["spectre"]

            plot(vitesse, courbes, zoom=zoom, labelList=labels, animation=False, ylim=[-1e6, 1.4e7],
                 xlabel='Speed [m/s]', ylabel='Amplitude (linéaire)', title="Amplitude du spectre du signal\nsuperposé avec le sinus cardinal théorique")

        sommeSpectrogrammes /= t / 0.05
        vitesse = np.linspace(-self.Vmax, self.Vmax, sommeSpectrogrammes.size)

        sinc = self.calculerSinc(vitesse, sommeSpectrogrammes, N)

        if sincCourbe:
            courbes, labels = [sommeSpectrogrammes, sinc], ["spectre", "sinus cardinal"]
        else:
            courbes, labels = [sommeSpectrogrammes], ["spectre"]

        plot(vitesse, courbes, zoom=zoom, labelList=labels, animation=False, ylim=[0, 1.2e7],
             xlabel='Speed [m/s]', ylabel='Amplitude (linéaire)',
             title="Amplitude moyenne du spectre du signal\n(en retirant la moyenne du signal temporel)")

    def afficherParametresRadar(self):
        """
        Méthode affichant les paramètres du radar.
        """
        parametresToutesFrames = self.data.RPRM
        parametres = list(dict.fromkeys(parametresToutesFrames))[0]
        print(parametres)

        parametres = np.frombuffer(parametres, dtype=np.uint16)
        param1 = parametres[0]
        param2 = parametres[1]
        param3 = parametres[2]
        param4 = parametres[3]

        print('\nParamètres du radar')
        print('\nRadar initial delay [clk] : {}'.format(param1))
        print('Radar start frequency [MHz]  : {}'.format(param2))
        print('Radar bandwidth [MHz] : {}'.format(param3))
        print('Radar receiver gain [dB] : {}\n'.format(param4))

    def afficherParametresProcesseur(self):
        """
        Méthode affichant les paramètres du processeur.
        """
        parametresToutesFrames = self.data.PPRM
        parametres = list(dict.fromkeys(parametresToutesFrames))[0]

        param1 = int.from_bytes(parametres[0:4], "little")
        param3 = int.from_bytes(parametres[8:10], "little")
        param4 = int.from_bytes(parametres[10:12], "little")
        param5 = struct.unpack("f", parametres[12:16])[0]
        param6 = int.from_bytes(parametres[16:18], "little")
        param7 = int.from_bytes(parametres[18:20], "little")
        param8 = int.from_bytes(parametres[20:22], "little")
        param9 = int.from_bytes(parametres[22:24], "little")
        param10 = int.from_bytes(parametres[24:26], "little")
        param12 = int.from_bytes(parametres[28:30], "little")
        param13 = int.from_bytes(parametres[30:32], "little")
        param14 = int.from_bytes(parametres[32:34], "little")
        param15 = int.from_bytes(parametres[34:36], "little")
        param16 = int.from_bytes(parametres[36:38], "little")
        param17 = int.from_bytes(parametres[38:40], "little", signed=True)
        param18 = int.from_bytes(parametres[40:42], "little")
        param19 = int.from_bytes(parametres[42:44], "little")
        param20 = int.from_bytes(parametres[44:46], "little")
        param22 = struct.unpack("f", parametres[48:52])[0]
        param23 = struct.unpack("f", parametres[52:56])[0]

        print('\nParamètres du processeur')
        print('\nProcessor peak detection threshold : {}'.format(param1))
        print('Processor maximum number of peaks : {}'.format(param3))
        print('Processor background update rate : {}'.format(param4))
        print('Processor range compensation for threshold : {}'.format(param5))
        print('Processor peak detection minimum range [bin] : {}'.format(param6))
        print('Processor peak detection maximum range [bin] : {}'.format(param7))
        print('Processor peak detection minimum speed [bin] : {}'.format(param8))
        print('Processor peak detection maximum speed [bin] : {}'.format(param9))
        print('Processor smooth mean range-doppler map : {}'.format(param10))
        print('Processor maximum number of tracks to report : {}'.format(param12))
        print('Processor maximum range jitter [bin] : {}'.format(param13))
        print('Processor maximum speed jitter [bin]  : {}'.format(param14))
        print('Processor minimum track life [frame] : {}'.format(param15))
        print('Processor maximum track life [frame] : {}'.format(param16))
        print('Direction error threshold [° x 100] : {}'.format(param17))
        print('Processor length of history for tracking [frame] : {}'.format(param18))
        print('Processor report stationary objects : {}'.format(param19))
        print('Processor assume constant speed for tracking : {}'.format(param20))
        print('Range scaling factor [m] : {}'.format(param22))
        print('Speed scaling factor [m/s] : {}\n'.format(param23))

    def trouverResolution(self):
        """
        Méthode retrouvant les résolutions en distance et en vitesse du radar, à partir des paramètres du processeur.
        """
        parametresToutesFrames = self.data.PPRM
        parametres = list(dict.fromkeys(parametresToutesFrames))[0]

        param22 = struct.unpack("f", parametres[48:52])[0]
        param23 = struct.unpack("f", parametres[52:56])[0]

        self.dR = param22
        self.Rmax = 255 * param22
        self.dV = param23
        self.Vmax = 127 * param23

    def supprimerZeroDoppler(self, signal, N) -> np.ndarray:
        """
        Méthode supprimant le sinus cardinal se situant au zéro doppler.

        :param signal: signal dont on souhaite retirer le sinus cardinal
        :param N: nombre de points fréquentiels pour le slow time du signal (largeur du signal)
        :return: signal dont le sinus cardinal a été retiré
        """
        vitesse = np.linspace(-self.Vmax, self.Vmax, signal.size)
        sinc = self.calculerSinc(vitesse, signal, N)

        signal -= sinc

        return signal

    def determinerSinc(self) -> (float, float):
        """
        Méthode déterminant le déphasage et la période du sinus cardinal correspondant au zéro-doppler pour le signal
        traité.

        :return: paramètres du sinus cardinal
        """
        sommeSpectrogrammes = None

        t = 0

        for signalRecu in self.signalRecuMean:

            if t >= 5:
                break
            t += 0.05

            signalRecu, L, N = transformeesDeFourierRangeDoppler(signalRecu, True)
            signalRecu = np.abs(signalRecu)

            indiceRange = np.argmax(np.quantile(signalRecu[5:250, 100:150], 0.60, axis=1)) + 5

            signalRange = signalRecu[indiceRange]

            if sommeSpectrogrammes is None:
                sommeSpectrogrammes = signalRange
            else:
                sommeSpectrogrammes += signalRange

        vitesse = np.linspace(-self.Vmax, self.Vmax, sommeSpectrogrammes.size)

        pred = np.inf
        PZ = np.argmax(sommeSpectrogrammes)
        L = PZ

        while pred > sommeSpectrogrammes[PZ]:
            pred = sommeSpectrogrammes[PZ]
            PZ += 1

        PZ -= 1
        SZP = PZ * 2 - L
        SZN = 3 * L - 2 * PZ

        secondZeroPositif = vitesse[np.argmin(sommeSpectrogrammes[SZP - 2:SZP + 2]) + SZP - 2]
        secondZeroNegatif = vitesse[np.argmin(sommeSpectrogrammes[SZN - 2:SZN + 2]) + SZN - 2]

        T = (secondZeroPositif - secondZeroNegatif) / 2
        Dv = (secondZeroPositif + secondZeroNegatif) / 2

        return T, Dv

    def calculerSinc(self, vitesse, signal, N) -> np.ndarray:
        """
        Méthode calculant un sinus cardinal à partir d'un vecteur vitesse et d'un signal.

        :param np.ndarray vitesse: vecteur vitesse correspondant à l'abscisse du sinus cardinal
        :param np.ndarray signal: signal représentant un sinus cardinal
        :param int N: nombre de points fréquentiels pour le slow time du signal (largeur du signal)
        :return: sinus cardinal
        """
        maxS, limS = np.max(signal), np.mean(signal[4 * N // 5:])
        return np.abs(np.sinc(2 * (vitesse - self.Dv) / self.T) * (maxS - limS))

    def chirpMoyen(self, i):
        """
        Méthode calculant le chirp moyen sur une frame et le dupliquant pour le mettre sous la forme d'une matrice
        fast-time/ slow time.

        :param i: indice de la frame dont on doit calculer le chirp moyen
        :return: matrice fast-time/ slow time contenant le chirp moyen dupliqué 256 fois
        """
        a = np.array(self.signalRecuMean[i])
        a = np.mean(a, axis=1).reshape(-1,)
        a = np.repeat(a, 256).reshape(256, 256)
        return a


if __name__ == '__main__':
    traitement = TraitementData("Data/Record8/mavic_no_pload_static_1.dat")

    # traitement.afficherChirp(0, 1)
    # traitement.afficherParametresProcesseur()
    # traitement.afficherParametresRadar()
    # traitement.afficherRangeDoppler(zeroPadding=True, zeroDoppler=False)
    # traitement.test(dist=242, zeroPadding=True, zeroDoppler=True, zoom=False, sincCourbe=False)
    traitement.afficherDopplerTime(zeroPadding=True, zeroDoppler=False, dist=231, methodeChirpMoyen=True)
    # traitement.afficherSpectrogrammeSpeed(zeroPadding=True, zoom=True, dist=242)
