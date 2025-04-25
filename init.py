import subprocess
import os
import signal
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get port from environment variables with a default fallback
PORT = int(os.getenv("WEBSOCKET_PORT", "3001"))

def run_ngrok(port):
    """Run ngrok with specified URL and port"""
    try:
        command = ["ngrok", "http", f"--url=literally-maximum-bison.ngrok-free.app", str(port)]
        print(f"Starting ngrok on port {port}...")
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        return process
    except Exception as e:
        print(f"Error starting ngrok: {str(e)}")
        return None

def run_clash():
    """Run clash with config directory"""
    try:
        config_path = os.path.expanduser("~/.config/clash")
        command = ["clash", "-d", config_path]
        print(f"Starting Clash with config directory: {config_path}")
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        return process
    except Exception as e:
        print(f"Error starting clash: {str(e)}")
        return None

def monitor_processes(clash_process, ngrok_process):
    """Monitor both processes and their output"""
    try:
        while True:
            # Print clash output
            clash_output = clash_process.stdout.readline()
            if clash_output:
                print("Clash:", clash_output.strip())
                
            # Print ngrok output
            ngrok_output = ngrok_process.stdout.readline()
            if ngrok_output:
                print("Ngrok:", ngrok_output.strip())
            
            # Check if either process has ended
            if clash_process.poll() is not None or ngrok_process.poll() is not None:
                print("One of the services has stopped unexpectedly")
                break
            
            time.sleep(0.1)  # Prevent CPU overuse
            
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        clash_process.terminate()
        ngrok_process.terminate()

def main():
    # Start both processes
    ngrok_process = run_ngrok(PORT)
    time.sleep(5)
    clash_process = run_clash()
    if not clash_process or not ngrok_process:
        print("Failed to start services")
        sys.exit(1)
    
    print(f"Starting services with port: {PORT}")
    
    # Handle Ctrl+C gracefully
    def signal_handler(signum, frame):
        print("\nStopping services...")
        clash_process.terminate()
        ngrok_process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Monitor both processes
    monitor_processes(clash_process, ngrok_process)

if __name__ == "__main__":
    main()