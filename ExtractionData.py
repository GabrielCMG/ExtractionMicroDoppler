class ExtractionData:
    """
    Classe contenant la méthode qui permet d'extraire les données radar d'un fichier issu du radar K-MD2.
    """

    def __init__(self, fileName):
        self._file = open(fileName, "rb")
        self._data = self._file.read()
        self.nombreFrame = 0
        self.RPRM = []
        self.PPRM = []
        self.RADC = []
        self.RMRD = []
        self.PDAT = []
        self.DONE = []
        self.lectureFichier()

    def lectureFichier(self):
        """
        Méthode extrayant les différentes informations du fichier contenant les données radar. Ces données peuvent
        correspondre, par exemple, aux données brutes du radar, aux paramètres du radar ou aux paramètres du processeur.
        """

        i = 0

        while i < len(self._data):
            # Lecture de l'entête
            header = self._data[i:i + 4].decode('utf-8')

            # Lecture de la longueur du message
            len_message = int.from_bytes(self._data[i + 4:i + 8], "little")

            # Lecture du message
            message = b''
            if len_message > 0:
                message = self._data[i + 8:i + 8 + len_message]

            # Incrémentation de l'indice vers la valeur correspondant à l'entête suivant
            i += 8 + len_message

            match header:
                case 'RPRM':
                    self.RPRM.append(message)

                case 'PPRM':
                    self.PPRM.append(message)

                case 'RADC':
                    self.RADC.append(message)

                case 'RMRD':
                    self.RMRD.append(message)

                case 'PDAT':
                    self.PDAT.append(message)

                case 'DONE':
                    self.nombreFrame += 1
