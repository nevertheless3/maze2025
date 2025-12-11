import serial
import time
import multiprocessing as mp
import os
import psutil
from threading import Thread


class SerialHandler:
    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.serial_send_queue = mp.Queue()
        self.serial_recv_queue = mp.Queue()
        self.serial_send_thread = None
        self.serial_recv_thread = None
        self.serial_process = None

    def crc16(self, data: bytes) -> int:
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001  # reversed poly 0x8005
                else:
                    crc >>= 1
        return crc & 0xFFFF

    def build_frame(self, payload: bytes) -> bytes:
        length = len(payload)
        crc = self.crc16(bytes([length]) + payload)

        frame = bytearray()
        frame.append(0xAA)  # start
        frame.append(length)  # length
        frame.extend(payload)  # data
        frame.append(crc & 0xFF)  # CRC low
        frame.append((crc >> 8) & 0xFF)  # CRC high
        frame.append(0x55)  # end
        return bytes(frame)

    def parse_frame(self, frame: bytes) -> bytes | None:
        if len(frame) < 5:
            return None
        if frame[0] != 0xAA or frame[-1] != 0x55:
            return None

        length = frame[1]
        if length + 5 != len(frame):
            return None

        payload = frame[2:2 + length]
        received_crc = frame[2 + length] | (frame[3 + length] << 8)
        calc_crc = self.crc16(bytes([length]) + payload)

        if received_crc != calc_crc:
            return None
        return payload

    def serial_process_run(self):
        self.serial_send_thread = Thread(target=self.serial_send_loop, daemon=True)
        self.serial_recv_thread = Thread(target=self.serial_recv_loop, daemon=True)

        self.serial_send_thread.start()
        self.serial_recv_thread.start()

        try:
            self.serial_send_thread.join()
            self.serial_recv_thread.join()
        except KeyboardInterrupt:
            pass

    def serial_send_loop(self):
        last_time_sent = 0
        while True:
            if not self.serial_send_queue.empty():
                data = self.serial_send_queue.get()
            else:
                this_time = time.time() - self.boot_time
                this_time = round(this_time)
                if this_time == last_time_sent:
                    time.sleep(1e-3)
                    continue
                data = f"T{this_time}"
                last_time_sent = this_time

            data = str(data) + "\n"
            # If you putted print here and it had an exterme delay, don't worry! The uart is sending it correctly, but I don't know why this delay happens.
            # The esp will receive it correctly on time :)
            frame = self.build_frame(data.encode('ascii'))
            self.ser.write(frame)
            # print("Sent data", frame)
            self.ser.flush()

    def serial_recv_loop(self):
        while True:
            if self.ser.in_waiting > 0:
                data = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if data:
                    self.serial_recv_queue.put(data)

    def config(self, boot_time):
        self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        self.serial_process = mp.Process(target=self.serial_process_run)
        self.boot_time = boot_time

    def run_serial(self):
        self.serial_process.start()

    def join_serial(self):
        try:
            self.serial_process.join()
        except KeyboardInterrupt:
            pass

    def stop_serial(self):
        # self.serial_send_thread.stop()
        # self.serial_recv_thread.stop()
        # self.serial_process.stop()
        self.ser.close()
        print("[INFO] Serial closed")

    def write(self, data):
        if not data:
            return
        self.serial_send_queue.put(str(data))

    def read(self):
        if not self.serial_recv_queue.empty():
            return str(self.serial_recv_queue.get())
        return None


if __name__ == "__main__":
    serial_handler = SerialHandler('/dev/serial0')
    serial_handler.config(0)
    serial_handler.run_serial()
    serial_handler.join_serial()  # Only if there is no more code (except for other tasks 'join' command) after this