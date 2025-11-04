import socket
import base64
import time
from picamera2 import Picamera2
import os
import threading
import queue
import subprocess # <--- Used to call espeak

# ====== Configuration ======
SERVER_IP = "192.168.160.130"  # !! IMPORTANT: Change to your server's IP
PORT = 5050
CAPTURE_INTERVAL = 0.5  # seconds between captures

# ====== Initialize Camera ======
try:
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(main={"size": (640, 480)}))
    cam.start()
    print("[CLIENT] ✅ Camera initialized")
except Exception as e:
    print(f"[CLIENT] ❌ Failed to initialize camera: {e}")
    exit()

# ====== Text-to-Speech (using espeak) ======
# This queue holds text for the TTS thread to speak
tts_queue = queue.Queue()

def speak_worker():
    """
    [THREAD 3 - Dedicated TTS Thread]
    Waits for text in the queue and speaks it using espeak.
    """
    print("[CLIENT TTS] ✅ TTS worker started.")
    while True:
        try:
            text = tts_queue.get() # Blocks until an item is available
            if text is None:
                break # Stop signal
            
            print(f"[CLIENT TTS] 🔊 Speaking: {text}")
            
            # Call espeak in a subprocess.
            # stdout/stderr are hidden to keep the console clean.
            subprocess.run(
                ["espeak", text], 
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            tts_queue.task_done()
        except Exception as e:
            print(f"[CLIENT TTS] ❌ espeak Error: {e}")

# === ROBUST SOCKET FUNCTIONS ===
def send_msg(conn, msg_bytes):
    """Safely send a message with an 8-byte length prefix."""
    try:
        data_len = f"{len(msg_bytes):8}".encode()
        conn.sendall(data_len + msg_bytes)
    except Exception as e:
        if not stop_event.is_set():
            print(f"[CLIENT SEND] ❌ Error sending message: {e}")
        raise # Raise error to stop the thread

def recv_msg(conn):
    """Safely receive a message with an 8-byte length prefix."""
    try:
        data_len_bytes = b""
        while len(data_len_bytes) < 8:
            packet = conn.recv(8 - len(data_len_bytes))
            if not packet:
                return None
            data_len_bytes += packet
        
        data_len = int(data_len_bytes.decode().strip())
        
        data = b""
        while len(data) < data_len:
            packet = conn.recv(data_len - len(data))
            if not packet:
                return None
            data += packet
            
        return data
    except Exception as e:
        if not stop_event.is_set():
            print(f"[CLIENT RECV] ❌ Error receiving message: {e}")
        return None

# === THREADING FUNCTIONS ===

def capture_and_send_loop(conn, cam, stop_event):
    """
    [THREAD 1]
    Captures and sends images at a fixed interval.
    """
    while not stop_event.is_set():
        try:
            image_path = "/tmp/frame.jpg"
            cam.capture_file(image_path)

            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            data_bytes = image_data.encode()
            send_msg(conn, data_bytes)
            print("[CLIENT SEND] 📤 Sent frame")
            
            time.sleep(CAPTURE_INTERVAL)

        except Exception as e:
            if not stop_event.is_set():
                print(f"[CLIENT SEND] ❌ Thread error: {e}")
            stop_event.set() # Signal other thread to stop

def receive_loop(conn, stop_event):
    """
    [THREAD 2]
    Listens for responses and adds them to the TTS queue.
    """
    while not stop_event.is_set():
        try:
            response_bytes = recv_msg(conn)
            if response_bytes is None:
                if not stop_event.is_set():
                    print("[CLIENT RECV] ❌ Server disconnected.")
                stop_event.set()
                break
                
            response = response_bytes.decode()
            print(f"[CLIENT RECV] 📥 Server response: {response}")

            # Add the response to the queue for the speak_worker to handle
            tts_queue.put(response)

        except Exception as e:
            if not stop_event.is_set():
                print(f"[CLIENT RECV] ❌ Thread error: {e}")
            stop_event.set()

# ====== Connect to Server ======
print("[CLIENT] 🔌 Connecting to server...")
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client.connect((SERVER_IP, PORT))
    print("[CLIENT] ✅ Connected to server")
except Exception as e:
    print(f"[CLIENT] ❌ Connection failed: {e}")
    cam.stop()
    exit()

# ====== Main Thread: Start and Manage Loops ======
stop_event = threading.Event()

# Create threads
send_thread = threading.Thread(target=capture_and_send_loop, args=(client, cam, stop_event))
recv_thread = threading.Thread(target=receive_loop, args=(client, stop_event))
tts_thread = threading.Thread(target=speak_worker)

# Start threads
print("[CLIENT] Starting threads...")
send_thread.start()
recv_thread.start()
tts_thread.start()

try:
    # Keep the main thread alive to catch KeyboardInterrupt
    while not stop_event.is_set():
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n[CLIENT] 🛑 Stopping client...")
finally:
    # Signal all threads to stop
    stop_event.set()
    tts_queue.put(None) # Send stop signal to TTS thread
    
    try:
        # A hard shutdown to unblock the client.recv() call
        client.shutdown(socket.SHUT_RDWR) 
        client.close()
    except OSError:
        pass # Socket already closed
    
    # Wait for threads to finish
    send_thread.join()
    recv_thread.join()
    tts_thread.join()
    
    cam.stop()
    print("[CLIENT] 🔌 Disconnected")