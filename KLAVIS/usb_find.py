import os
import sys
import platform
import subprocess
import time
import threading
import json
import signal
from datetime import datetime, timedelta

TARGET = "VID_1209&PID_BEEE"

print("Waiting for Solo key...")

gui_open = False
last_trigger_time = 0
DEBOUNCE_SECONDS = 3

stop_event = threading.Event()


# ---------------------------------------------------
# FORCE CLEAN EXIT (Ctrl+C FIX)
# ---------------------------------------------------
def handle_exit(signum, frame):
    print("\nCtrl+C detected. Shutting down...")
    stop_event.set()
    sys.exit(0)


signal.signal(signal.SIGINT, handle_exit)


# ---------------------------------------------------
# Certificate / persistence setup
# ---------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CERT_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "certificates")
)

os.makedirs(CERT_DIR, exist_ok=True)

CERT_FILE = os.path.join(CERT_DIR, "solo_key_certificate.json")

CERT_EXPIRY_DAYS = 90  # ~3 months


def load_certificate():
    if not os.path.exists(CERT_FILE):
        return {}

    try:
        with open(CERT_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_certificate(data):
    try:
        with open(CERT_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Failed to save certificate: {e}")


cert_data = load_certificate()


def is_certificate_expired(first_seen_str):
    first_seen = datetime.fromisoformat(first_seen_str)
    return datetime.now() > first_seen + timedelta(days=CERT_EXPIRY_DAYS)


def register_first_seen(device_id):
    if device_id not in cert_data:
        cert_data[device_id] = {
            "first_seen": datetime.now().isoformat()
        }
        save_certificate(cert_data)


def requires_enrollment(device_id):
    if device_id not in cert_data:
        return True

    return is_certificate_expired(cert_data[device_id]["first_seen"])


# ---------------------------------------------------
# PATHS
# ---------------------------------------------------
SOFTWARE_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "Software/Adaptive_Enrollment_Update")
)

if SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, SOFTWARE_DIR)


# ---------------------------------------------------
# GUI
# ---------------------------------------------------
GUI_MODE = sys.argv[1] if len(sys.argv) > 1 else "user"


def _get_main():
    if GUI_MODE == "developer":
        from developer_gui import main
    else:
        from user_gui import main
    return main


def show_gui():
    global gui_open

    if gui_open:
        return

    gui_open = True

    def run_gui():
        global gui_open
        try:
            print(f"Launching KLAVIS GUI ({GUI_MODE} mode)...")
            _get_main()()
        except Exception as e:
            print(f"Failed to launch GUI: {e}")
        finally:
            gui_open = False

    if platform.system() == "Darwin":
        run_gui()
    else:
        threading.Thread(target=run_gui, daemon=True).start()


def show_enrollment():
    global gui_open

    if gui_open:
        return

    gui_open = True

    def run_enrollment():
        global gui_open
        try:
            print(f"Launching ENROLLMENT ({GUI_MODE} mode)...")
            _get_main()()
        except Exception as e:
            print(f"Failed to launch enrollment: {e}")
        finally:
            gui_open = False

    if platform.system() == "Darwin":
        run_enrollment()
    else:
        threading.Thread(target=run_enrollment, daemon=True).start()


def show_enrollment():
    global gui_open

    if gui_open:
        return

    gui_open = True

    def run_enrollment():
        global gui_open
        try:
            print("Launching ENROLLMENT (certificate expired)...")
            from main_gui import main as enrollment_main
            enrollment_main(enrollment_mode=True)
        except Exception as e:
            print(f"Failed to launch enrollment: {e}")
        finally:
            gui_open = False

    if platform.system() == "Darwin":
        run_enrollment()
    else:
        threading.Thread(target=run_enrollment, daemon=True).start()


# ---------------------------------------------------
# DEBOUNCE
# ---------------------------------------------------
def should_trigger():
    global last_trigger_time

    now = time.time()

    if now - last_trigger_time < DEBOUNCE_SECONDS:
        return False

    last_trigger_time = now
    return True


# ---------------------------------------------------
# WINDOWS (FIXED)
# ---------------------------------------------------
def watch_windows():
    import wmi

    c = wmi.WMI()
    watcher = c.Win32_PnPEntity.watch_for("creation")

    while not stop_event.is_set():
        try:
            device = watcher(timeout_ms=500)  # ✅ prevents Ctrl+C freeze

            if TARGET in device.PNPDeviceID and should_trigger():
                print("Solo key connected!")

                register_first_seen(TARGET)

                if requires_enrollment(TARGET):
                    show_enrollment()
                else:
                    show_gui()

        except wmi.x_wmi_timed_out:
            continue
        except Exception:
            continue


# ---------------------------------------------------
# LINUX
# ---------------------------------------------------
def watch_linux():
    import pyudev

    context = pyudev.Context()
    monitor = pyudev.Monitor.from_netlink(context)
    monitor.filter_by(subsystem='usb')

    for device in monitor:
        if stop_event.is_set():
            break

        if device.action == 'add':
            vid = device.get('ID_VENDOR_ID')
            pid = device.get('ID_MODEL_ID')

            if vid and pid:
                formatted = f"VID_{vid.upper()}&PID_{pid.upper()}"

                if formatted == TARGET and should_trigger():
                    print("Solo key connected!")

                    register_first_seen(TARGET)

                    if requires_enrollment(TARGET):
                        show_enrollment()
                    else:
                        show_gui()


# ---------------------------------------------------
# macOS (FIXED TIMEOUT)
# ---------------------------------------------------
def watch_macos():
    was_connected = False

    while not stop_event.is_set():
        try:
            output = subprocess.check_output(
                ["ioreg", "-p", "IOUSB", "-l", "-w", "0"],
                text=True,
                timeout=2  #  prevents hang on Ctrl+C
            )

            is_connected = (
                "Solo 2 Security Key" in output or
                ('"idVendor" = 4617' in output and '"idProduct" = 48878' in output)
            )

            if is_connected and not was_connected:
                if should_trigger():
                    print("Solo key connected!")

                    register_first_seen(TARGET)

                    if requires_enrollment(TARGET):
                        show_enrollment()
                    else:
                        show_gui()

            was_connected = is_connected
            time.sleep(2)

        except subprocess.TimeoutExpired:
            continue


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    system = platform.system()
    print(f"Running USB watcher on {system}")

    try:
        if system == "Windows":
            watch_windows()
        elif system == "Linux":
            watch_linux()
        elif system == "Darwin":
            watch_macos()
        else:
            print(f"Unsupported OS: {system}")

    finally:
        stop_event.set()
        print("Watcher stopped.")


if __name__ == "__main__":
    main()