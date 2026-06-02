"""
socket_stress.py

Internal Docker-network traffic generator for reproducible telemetry tests.

Use only against the local Docker training nodes:
  node0, node1, node2

Modes:
  server: listens on a TCP port and discards incoming data
  client: opens one or more TCP streams and sends data for a fixed duration

Example:
  python3 tools/socket_stress.py server --port 23456
  python3 tools/socket_stress.py client --target node1 --port 23456 --duration 120 --streams 4 --mbps 80
"""

import argparse
import socket
import threading
import time


DEFAULT_PORT = 23456
CHUNK_SIZE = 64 * 1024


def run_server(host: str, port: int):
    total_bytes = 0
    total_lock = threading.Lock()
    stop = threading.Event()

    def reporter():
        nonlocal total_bytes
        last_total = 0
        last_time = time.time()

        while not stop.is_set():
            time.sleep(2)
            now = time.time()

            with total_lock:
                current_total = total_bytes

            elapsed = now - last_time
            delta = current_total - last_total
            mbps = (delta * 8) / elapsed / 1_000_000 if elapsed > 0 else 0

            print(
                f"[server] received={current_total / 1_000_000:.2f} MB "
                f"rate={mbps:.2f} Mbps",
                flush=True,
            )

            last_total = current_total
            last_time = now

    def handle_client(conn, addr):
        nonlocal total_bytes
        print(f"[server] connection from {addr}", flush=True)

        try:
            while True:
                data = conn.recv(CHUNK_SIZE)
                if not data:
                    break

                with total_lock:
                    total_bytes += len(data)
        except Exception as e:
            print(f"[server] client error from {addr}: {e}", flush=True)
        finally:
            conn.close()
            print(f"[server] connection closed from {addr}", flush=True)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(64)

    print(f"[server] listening on {host}:{port}", flush=True)

    threading.Thread(target=reporter, daemon=True).start()

    try:
        while True:
            conn, addr = server.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        print("[server] stopping", flush=True)
    finally:
        stop.set()
        server.close()


def run_client(target: str, port: int, duration: float, streams: int, mbps: float | None):
    print(
        f"[client] target={target}:{port} duration={duration}s "
        f"streams={streams} mbps={mbps if mbps else 'unlimited'}",
        flush=True,
    )

    deadline = time.time() + duration
    payload = b"\0" * CHUNK_SIZE
    total_bytes = 0
    total_lock = threading.Lock()

    # If mbps is specified, divide the bandwidth budget across streams.
    bytes_per_second_total = (mbps * 1_000_000 / 8) if mbps else None
    bytes_per_second_per_stream = (
        bytes_per_second_total / streams if bytes_per_second_total else None
    )

    def worker(worker_id: int):
        nonlocal total_bytes

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((target, port))
            print(f"[client:{worker_id}] connected", flush=True)

            sent_this_second = 0
            second_window = time.time()

            while time.time() < deadline:
                sock.sendall(payload)

                with total_lock:
                    total_bytes += len(payload)

                if bytes_per_second_per_stream:
                    sent_this_second += len(payload)
                    now = time.time()

                    if now - second_window >= 1.0:
                        sent_this_second = 0
                        second_window = now
                    elif sent_this_second >= bytes_per_second_per_stream:
                        sleep_time = 1.0 - (now - second_window)
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        sent_this_second = 0
                        second_window = time.time()

            sock.close()
            print(f"[client:{worker_id}] done", flush=True)

        except Exception as e:
            print(f"[client:{worker_id}] error: {e}", flush=True)

    threads = []
    start = time.time()

    for i in range(streams):
        t = threading.Thread(target=worker, args=(i,), daemon=True)
        t.start()
        threads.append(t)

    while time.time() < deadline:
        time.sleep(2)
        elapsed = time.time() - start

        with total_lock:
            current_total = total_bytes

        avg_mbps = (current_total * 8) / elapsed / 1_000_000 if elapsed > 0 else 0
        print(
            f"[client] sent={current_total / 1_000_000:.2f} MB "
            f"avg_rate={avg_mbps:.2f} Mbps",
            flush=True,
        )

    for t in threads:
        t.join(timeout=2)

    elapsed = time.time() - start
    with total_lock:
        final_total = total_bytes

    avg_mbps = (final_total * 8) / elapsed / 1_000_000 if elapsed > 0 else 0

    print(
        f"[client] finished sent={final_total / 1_000_000:.2f} MB "
        f"avg_rate={avg_mbps:.2f} Mbps",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Socket-based network stress tool")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    server_parser = subparsers.add_parser("server")
    server_parser.add_argument("--host", default="0.0.0.0")
    server_parser.add_argument("--port", type=int, default=DEFAULT_PORT)

    client_parser = subparsers.add_parser("client")
    client_parser.add_argument("--target", required=True)
    client_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    client_parser.add_argument("--duration", type=float, default=120)
    client_parser.add_argument("--streams", type=int, default=4)
    client_parser.add_argument(
        "--mbps",
        type=float,
        default=None,
        help="Optional total Mbps limit across all streams. Omit for unlimited.",
    )

    args = parser.parse_args()

    if args.mode == "server":
        run_server(args.host, args.port)
    elif args.mode == "client":
        run_client(args.target, args.port, args.duration, args.streams, args.mbps)


if __name__ == "__main__":
    main()
