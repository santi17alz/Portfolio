# features.py

# Keys to completely ignore in feature extraction
IGNORED_KEYS = {
    'Key.backspace', 'Key.shift', 'Key.shift_r',
    'Key.ctrl', 'Key.ctrl_r', 'Key.alt', 'Key.alt_r',
    'Key.cmd', 'Key.enter', 'Key.tab', 'Key.caps_lock',
    'BackSpace', 'Shift_L', 'Shift_R', 'Control_L', 'Control_R',
    'Alt_L', 'Alt_R', 'Meta_L', 'Meta_R', 'Return', 'Tab'
}

def extract_features(events):
    features  = []
    up_times  = {}

    # Filter out ignored keys from all events
    clean_events = [
        (k, e, t) for k, e, t in events
        if k not in IGNORED_KEYS
    ]

    for key, event_type, timestamp in clean_events:
        if event_type == 'up':
            up_times[key] = timestamp

    # Only look at keydown events in order
    down_events = [(k, t) for k, e, t in clean_events if e == 'down']

    for i in range(len(down_events) - 1):
        current_key, current_down = down_events[i]
        next_key,    next_down    = down_events[i + 1]

        if current_key not in up_times:
            continue

        current_up  = up_times[current_key]
        dwell_time  = current_up - current_down
        flight_time = max(0.0, next_down - current_up)  # clip negatives

        features.append({
            'bigram': f"{current_key}-{next_key}",
            'dwell':  round(dwell_time, 4),
            'flight': round(flight_time, 4)
        })

    return features