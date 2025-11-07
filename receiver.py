import can

def receive_messages():
    with can.interface.Bus(channel='vcan0', bustype='socketcan') as bus:
        print("Listening for messages on vcan0 (Ctrl+C to stop)...")
        while True:
            message = bus.recv(1.0)  # timeout in seconds
            if message is not None:
                print(f"Received: {message}")

if __name__ == "__main__":
    try:
        receive_messages()
    except KeyboardInterrupt:
        print("\nReceiver stopped")
