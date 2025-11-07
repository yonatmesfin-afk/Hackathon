import can

def send_one():
    with can.interface.Bus(channel='vcan0', bustype='socketcan') as bus:
        msg = can.Message(
            arbitration_id=0x123,
            data=[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
            is_extended_id=False,
        )
        try:
            bus.send(msg)
            print(f"Message sent on {bus.channel_info}")
        except can.CanError as e:
            print(f"Message NOT sent: {e}")

if __name__ == "__main__":
    send_one()
