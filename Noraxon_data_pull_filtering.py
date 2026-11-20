import requests
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import math
from scipy.signal import butter, iirnotch, tf2sos, sosfilt, welch
import threading

SERVER_IP = "192.168.1.133"
SERVER_PORT = 8000

URL = f"http://{SERVER_IP}:{SERVER_PORT}/stream"

class NoraxonStreamClient:
    def __init__(self):
        self.url = URL
        self.rms = []
        self.raw_emg = []
        self.filtered_emg = []
        self.batch1=[]
        self.batch2=[]
        self.batch2_begin_flag = False
        self.filter_state = None
        self.fs = 2000.4
        self.rms_nofilt = []
        self.raw_emg_no_filt = []

        ##creating filters -- Bandpass at 20-500Hz and Notch filter from 50-60
        self.bandpass_sos = butter(
            4, 
            [20, 450], 
            btype = 'bandpass',
            fs = self.fs,
            output = 'sos'
        )
        self.bandstop_sos = butter(
            4, 
            [50, 60],
            btype = 'bandstop',
            fs = self.fs,
            output='sos'
        )

        self.filter_sos = np.vstack((self.bandpass_sos, self.bandstop_sos))
        self.filter_state = np.zeros((self.filter_sos.shape[0], 2))

    def receive_stream(self):
        print(f"Connecting to stream at {self.url}...")
        try:
            with requests.get(self.url, stream=True, timeout=(5, 3)) as response:  ##reading HTTP packets
                response.raise_for_status()  # Raise an error for bad responses
                print("Connected. Receiving data...")
                for line in response.iter_lines():
                    if not line: 
                        continue  # Skip empty lines
                    if line:
                        packet = json.loads(line.decode('utf-8')) 
                        #packet set up should have only sequence, timestamp, and emg will likely change with actual Noraxon equipment

                        sequence_number = packet.get("sequence")
                        timestamp = packet.get("timestamp")
                        emg = packet.get("emg")
                        print(f"Sequence: {sequence_number}, Timestamp: {timestamp}, EMG: {emg}")
                        self.process_packet(emg)
            print(self.rms)
        except requests.exceptions.RequestException as e:
            print(f"Error receiving stream: {e}")
    def process_packet(self, emg_packet):
        emg = np.asarray(emg_packet, dtype=float)
        n = len(emg)
        for i in range(n): 
            emg_point = emg[i]
            self.raw_emg.append(emg_point)
            if len(self.batch1) < 250:
                self.batch1.append(emg_point)
            else:
                self.process_batch(self.batch1)
                self.batch1 = []
            if len(self.batch1) == 125:
                self.batch2_begin_flag = True
            if self.batch2_begin_flag:
                self.batch2.append(emg_point)
            if len(self.batch2) > 250:
                self.process_batch(self.batch2)
                self.batch2_begin_flag = False
                self.batch2 = []

    def process_batch(self, batch):
        filtered_emg, self.filter_state = sosfilt(
            self.filter_sos, 
            batch, 
            zi=self.filter_state

        )
        sum_s_emg = 0
        n = len(filtered_emg)
        for i in range(n): 
            emg = filtered_emg[i]
            self.filtered_emg.append(emg)
            rect_emg = abs(emg)
            sum_s_emg += rect_emg ** 2 #rectifies and squares

        mean_sq = (1/n)*sum_s_emg
        rms = math.sqrt(mean_sq)
        self.rms.append(rms)
        print(rms)
        
        sum_s_emg_raw = 0
        n = len(batch)
        for i in range(n): 
            emg = batch[i]
            self.raw_emg_no_filt.append(emg)
            rect_emg = abs(emg)
            sum_s_emg_raw += rect_emg ** 2 #rectifies and squares

        mean_sq = (1/n)*sum_s_emg_raw
        rms = math.sqrt(mean_sq)
        self.rms_nofilt.append(rms)
        print(rms)


    def plot(self):
        plt.figure(figsize=(12,6))
        plt.plot(self.rms_nofilt, linewidth=0.8)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude (mV)')
        plt.show()
        
        plt.figure(figsize=(12,6))
        plt.plot(self.rms, linewidth=0.8)
        plt.show()

    def data_review(self):
        raw_frequencies, raw_psd = welch(
            np.asarray(self.raw_emg, dtype=float), 
            fs=self.fs,
            window='hann',
            nperseg=1024
        )
        filter_frequencies, filtered_psd = welch(
                    np.asarray(self.filtered_emg, dtype=float), 
                    fs=self.fs,
                    window='hann',
                    nperseg=1024
                )
        ##Raw
        plt.figure(figsize=(12, 5))
        plt.plot(raw_frequencies, raw_psd)

        plt.title('Power Spectral Density - Raw')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.xlim((0, 400))
        plt.gca().xaxis.set_major_locator(MultipleLocator(10))
        plt.grid(True)

        plt.tight_layout()
        plt.show()


        ##Filtered
        plt.figure(figsize=(12, 5))
        plt.plot(filter_frequencies, filtered_psd)

        plt.title('Power Spectral Density - Filtered')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.gca().xaxis.set_major_locator(MultipleLocator(10))
        plt.xlim((0, 400))
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # ##RMS
        # plt.figure(figsize=(12, 5))
        # plt.plot(rms_frequencies, rms_psd)

        # plt.title('Power Spectral Density - RMS + Rect')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Power Spectral Density')
        # plt.grid(True)

        # plt.tight_layout()
        # plt.show()

        


if __name__ == "__main__":
    client = NoraxonStreamClient()
    client.receive_stream()
    client.plot()
    client.data_review()