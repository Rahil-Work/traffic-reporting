# project/tracking/cleanup.py
from core.utils import debug_print

# Configurable thresholds
STALE_VEHICLE_TIMEOUT_SECONDS = 300 # 5 mins video time
MAX_COMPLETED_PATHS = 2000

def cleanup_tracking_data(vehicle_tracker, zone_tracker, current_frame_time):
    """Removes data for vehicles unseen for a long time."""
    stale_ids = []
    if hasattr(vehicle_tracker, 'last_seen_time'):
        for v_id, last_seen in list(vehicle_tracker.last_seen_time.items()):
            if current_frame_time - last_seen > STALE_VEHICLE_TIMEOUT_SECONDS:
                stale_ids.append(v_id)

    if not stale_ids: return 0
    debug_print(f"Cleanup: Found {len(stale_ids)} stale vehicle IDs.")
    count = 0
    for v_id in stale_ids:
        if vehicle_tracker.remove_vehicle_data(v_id): count += 1
        if zone_tracker: zone_tracker.remove_vehicle_data(v_id)

    # Cleanup internal cooldowns within zone/line objects
    if zone_tracker and hasattr(zone_tracker, 'cleanup_cooldowns'): zone_tracker.cleanup_cooldowns(current_frame_time)
    if zone_tracker and hasattr(zone_tracker, 'lines'):
        for line in zone_tracker.lines.values():
            if line and hasattr(line, 'cleanup_cooldowns'): line.cleanup_cooldowns(current_frame_time)

    # Cap completed paths
    if hasattr(vehicle_tracker, 'completed_paths') and len(vehicle_tracker.completed_paths) > MAX_COMPLETED_PATHS:
         removed = len(vehicle_tracker.completed_paths) - MAX_COMPLETED_PATHS
         vehicle_tracker.completed_paths = vehicle_tracker.completed_paths[-MAX_COMPLETED_PATHS:]
         debug_print(f"Cleanup: Capped completed_paths, removed {removed}.")

    debug_print(f"Cleanup: Removed data for {count} stale vehicles.")
    return count