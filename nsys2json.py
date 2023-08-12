import sqlite3
import argparse
import json
import re

from collections import defaultdict

_PID_TO_DEVICE = None

# Code adapted from https://github.com/ezyang/nvprof2json

parser = argparse.ArgumentParser(description='Convert nsight systems sqlite output to Google Event Trace compatible JSON.')
parser.add_argument('--filename')
parser.add_argument("-o", "--output", help="Output file name", required=True)
parser.add_argument("-t", "--activity-type", help="Type of activities shown. Default to all.", default=["kernel", "nvtx-kernel"], choices=['kernel', 'nvtx', "nvtx-kernel"], nargs="+")
parser.add_argument("--nvtx-event-prefix", help="Filter NVTX events by their names' prefix.", type=str, nargs="*")
parser.add_argument("--nvtx-color-scheme", help="""Color scheme for NVTX events.
                                                   Accepts a dict mapping a string to one of chrome tracing colors.
                                                   Events with names containing the string will be colored.
                                                   E.g. {"send": "thread_state_iowait", "recv": "thread_state_iowait", "compute": "thread_state_running"}
                                                   For details of the color scheme, see links in https://github.com/google/perfetto/issues/208
                                                   """, type=json.loads, default={})
args = parser.parse_args()

def munge_time(t):
    """Take a timestamp from nsys (ns) and convert it into us (the default for chrome://tracing)."""
    # For strict correctness, divide by 1000, but this reduces accuracy.
    return t / 1000.

# For reference of the schema, see
# https://docs.nvidia.com/nsight-systems/UserGuide/index.html#exporter-sqlite-schema

def parse_cupti_kernel_events(conn: sqlite3.Connection, strings: dict, traceEvents: list = None):
    """
    Copied from the docs:
    CUPTI_ACTIVITY_KIND_KERNEL
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    streamId                    INTEGER   NOT NULL,                    -- Stream ID.
    correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    demangledName               INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Kernel function name w/ templates
    shortName                   INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Base kernel function name
    mangledName                 INTEGER,                               -- REFERENCES StringIds(id) -- Raw C++ mangled kernel function name
    launchType                  INTEGER,                               -- REFERENCES ENUM_CUDA_KRENEL_LAUNCH_TYPE(id)
    cacheConfig                 INTEGER,                               -- REFERENCES ENUM_CUDA_FUNC_CACHE_CONFIG(id)
    registersPerThread          INTEGER   NOT NULL,                    -- Number of registers required for each thread executing the kernel.
    gridX                       INTEGER   NOT NULL,                    -- X-dimension grid size.
    gridY                       INTEGER   NOT NULL,                    -- Y-dimension grid size.
    gridZ                       INTEGER   NOT NULL,                    -- Z-dimension grid size.
    blockX                      INTEGER   NOT NULL,                    -- X-dimension block size.
    blockY                      INTEGER   NOT NULL,                    -- Y-dimension block size.
    blockZ                      INTEGER   NOT NULL,                    -- Z-dimension block size.
    staticSharedMemory          INTEGER   NOT NULL,                    -- Static shared memory allocated for the kernel (B).
    dynamicSharedMemory         INTEGER   NOT NULL,                    -- Dynamic shared memory reserved for the kernel (B).
    localMemoryPerThread        INTEGER   NOT NULL,                    -- Amount of local memory reserved for each thread (B).
    localMemoryTotal            INTEGER   NOT NULL,                    -- Total amount of local memory reserved for the kernel (B).
    gridId                      INTEGER   NOT NULL,                    -- Unique grid ID of the kernel assigned at runtime.
    sharedMemoryExecuted        INTEGER,                               -- Shared memory size set by the driver.
    graphNodeId                 INTEGER,                               -- REFERENCES CUDA_GRAPH_EVENTS(graphNodeId)
    sharedMemoryLimitConfig     INTEGER                                -- REFERENCES ENUM_CUDA_SHARED_MEM_LIMIT_CONFIG(id)
    """
    per_device_kernel_events = defaultdict(list)
    for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        per_device_kernel_events[row["deviceId"]].append(row)
        if traceEvents is not None:
            event = {
                    "name": strings[row["shortName"]],
                    "ph": "X", # Complete Event (Begin + End event)
                    "cat": "cuda",
                    "ts": munge_time(row["start"]),
                    "dur": munge_time(row["end"] - row["start"]),
                    "tid": "Stream {}".format(row["streamId"]),
                    "pid": "Device {}".format(row["deviceId"]),
                    "args": {
                        # TODO: More
                        },
                    }
            traceEvents.append(event)
    return per_device_kernel_events

def link_pid_with_devices(conn: sqlite3.Connection):
    # map each pid to a device. assumes each pid is associated with a single device
    global _PID_TO_DEVICE
    if _PID_TO_DEVICE is None:
        pid_to_device = {}
        for row in conn.execute("SELECT DISTINCT deviceId, globalPid / 0x1000000 % 0x1000000 AS PID FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            assert row["PID"] not in pid_to_device, \
                f"A single PID ({row['PID']}) is associated with multiple devices ({pid_to_device[row['PID']]} and {row['deviceId']})."
            pid_to_device[row["PID"]] = row["deviceId"]
        _PID_TO_DEVICE = pid_to_device
    return _PID_TO_DEVICE

def parse_nvtx_events(conn: sqlite3.Connection, traceEvents: list = None, event_prefix=None, color_scheme={}):
    """
    Copied from the docs:
    NVTX_EVENTS
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER,                               -- Event end timestamp (ns).
    eventType                   INTEGER   NOT NULL,                    -- NVTX event type enum value. See docs for specifics.
    rangeId                     INTEGER,                               -- Correlation ID returned from a nvtxRangeStart call.
    category                    INTEGER,                               -- User-controlled ID that can be used to group events.
    color                       INTEGER,                               -- Encoded ARGB color value.
    text                        TEXT,                                  -- Optional text message for non registered strings.
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    endGlobalTid                INTEGER,                               -- Serialized GlobalId. See docs for specifics.
    textId                      INTEGER   REFERENCES StringIds(id),    -- StringId of the NVTX domain registered string.
    domainId                    INTEGER,                               -- User-controlled ID that can be used to group events.
    uint64Value                 INTEGER,                               -- One of possible payload value union members.
    int64Value                  INTEGER,                               -- One of possible payload value union members.
    doubleValue                 REAL,                                  -- One of possible payload value union members.
    uint32Value                 INTEGER,                               -- One of possible payload value union members.
    int32Value                  INTEGER,                               -- One of possible payload value union members.
    floatValue                  REAL,                                  -- One of possible payload value union members.
    jsonTextId                  INTEGER,                               -- One of possible payload value union members.
    jsonText                    TEXT                                   -- One of possible payload value union members.

    NVTX_EVENT_TYPES
    33 - NvtxCategory
    34 - NvtxMark
    39 - NvtxThread
    59 - NvtxPushPopRange
    60 - NvtxStartEndRange
    75 - NvtxDomainCreate
    76 - NvtxDomainDestroy
    """

    if event_prefix is None:
        match_text = ''
    else:
        match_text = " AND "
        if len(event_prefix) == 1:
            match_text += f"NVTX_EVENTS.text LIKE '{event_prefix}%'"
        else:
            match_text += "("
            for idx, prefix in enumerate(event_prefix):
                match_text += f"NVTX_EVENTS.text LIKE '{prefix}%'"
                if idx == len(event_prefix) - 1:
                    match_text += ")"
                else:
                    match_text += " OR "

    nvtx_events_per_device = defaultdict(list)
    pid_to_device = link_pid_with_devices(conn)
    # eventType 59 is NvtxPushPopRange, which corresponds to torch.cuda.nvtx.range apis
    for row in conn.execute(f"SELECT start, end, text, globalTid / 0x1000000 % 0x1000000 AS PID, globalTid % 0x1000000 AS TID FROM NVTX_EVENTS WHERE NVTX_EVENTS.eventType == 59{match_text};"):
        text = row['text']
        pid = row['PID']
        tid = row['TID']
        nvtx_events_per_device[pid_to_device[pid]].append(row)
        if traceEvents is not None:
            assert pid in pid_to_device, f"PID {pid} not found in the pid to device map."
            event = {
                    "name": text,
                    "ph": "X", # Complete Event (Begin + End event)
                    "cat": "nvtx",
                    "ts": munge_time(row["start"]),
                    "dur": munge_time(row["end"] - row["start"]),
                    "tid": "NVTX Thread {}".format(tid),
                    "pid": "Device {}".format(pid_to_device[pid]),
                    "args": {
                        # TODO: More
                        },
                    }
            if color_scheme:
                for key, color in color_scheme.items():
                    if re.search(key, text):
                        event["cname"] = color
                        break
            traceEvents.append(event)
    return nvtx_events_per_device

def parse_cuda_api_events(conn: sqlite3.Connection, strings: dict, traceEvents: list = None):
    """
    Copied from the docs:
    CUPTI_ACTIVITY_KIND_RUNTIME
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- CUDA event class enum value. See docs for specifics.
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- ID used to identify events that this function call has triggered.
    nameId                      INTEGER   NOT NULL   REFERENCES StringIds(id), -- StringId of the function name.
    returnValue                 INTEGER   NOT NULL,                    -- Return value of the function call.
    callchainId                 INTEGER   REFERENCES CUDA_CALLCHAINS(id) -- ID of the attached callchain.
    """
    pid_to_devices = link_pid_with_devices(conn)
    per_device_api_events = defaultdict(list)
    # event type 0 is TRACE_PROCESS_EVENT_CUDA_RUNTIME
    for row in conn.execute(f"SELECT start, end, globalTid / 0x1000000 % 0x1000000 AS PID, globalTid % 0x1000000 AS TID, correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME;"):
        text = strings[row['nameId']]
        pid = row['PID']
        tid = row['TID']
        correlationId = row['correlationId']
        per_device_api_events[pid_to_devices[pid]].append(row)
        if traceEvents is not None:
            event = {
                    "name": text,
                    "ph": "X", # Complete Event (Begin + End event)
                    "cat": "cuda_api",
                    "ts": munge_time(row["start"]),
                    "dur": munge_time(row["end"] - row["start"]),
                    "tid": "CUDA API Thread {}".format(tid),
                    "pid": "Device {}".format(pid_to_devices[pid]),
                    "args": {
                            "correlationId": correlationId,
                        },
                    }
            traceEvents.append(event)
    return per_device_api_events

def _find_overlapping_intervals(nvtx_events, cuda_api_events):
    mixed_events = []
    for nvtx_event in nvtx_events:
        start = nvtx_event["start"]
        end = nvtx_event["end"]
        mixed_events.append((start, 1, "nvtx", nvtx_event))
        mixed_events.append((end, -1, "nvtx", nvtx_event))
    for cuda_api_events in cuda_api_events:
        start = cuda_api_events["start"]
        end = cuda_api_events["end"]
        mixed_events.append((start, 1, "cuda_api", cuda_api_events))
        mixed_events.append((end, -1, "cuda_api", cuda_api_events))
    mixed_events.sort(key=lambda x: (x[0], x[1], x[2]))
    active_intervals = []
    result = defaultdict(list)
    for _, event_type, event_origin, orig_event in mixed_events:
        if event_type == 1:
            # start
            if event_origin == "nvtx":
                active_intervals.append(orig_event)
            else:
                for event in active_intervals:
                    result[event].append(orig_event)
        else:
            # end
            if event_origin == "nvtx":
                active_intervals.remove(orig_event)
    return result

def link_nvtx_events_to_kernel_events(strings: dict,
                                      pid_to_device: dict[int, int],
                                      nvtx_events_per_device: dict[int, list],
                                      cuda_api_events_per_device: dict[int, list],
                                      cuda_kernel_events_per_device: dict[int, list]):
    """
    Link NVTX events to cupti kernel events. This is done by first matching
    the nvtx ranges with CUDA API calls by timestamp. Then, retrieve the
    corresponding kernel events using the correlationId from CUDA API calls.
    """
    result = {}
    for device in pid_to_device.values():
        event_map = _find_overlapping_intervals(nvtx_events_per_device[device], cuda_api_events_per_device[device])
        correlation_id_map = defaultdict(dict)
        for cuda_api_event in cuda_api_events_per_device[device]:
            correlation_id_map[cuda_api_event["correlationId"]]["cuda_api"] = cuda_api_event
        for kernel_event in cuda_kernel_events_per_device[device]:
            correlation_id_map[kernel_event["correlationId"]]["kernel"] = kernel_event
        for nvtx_event, cuda_api_events in event_map.items():
            kernel_start_time = None
            kernel_end_time = None
            for cuda_api_event in cuda_api_events:
                if "kernel" not in correlation_id_map[cuda_api_event["correlationId"]]:
                    # other cuda api event, ignore
                    continue
                kernel_event = correlation_id_map[cuda_api_event["correlationId"]]["kernel"]
                if kernel_start_time is None or kernel_start_time > kernel_event["start"]:
                    kernel_start_time = kernel_event["start"]
                if kernel_end_time is None or kernel_end_time < kernel_event["end"]:
                    kernel_end_time = kernel_event["end"]
            if kernel_start_time is not None and kernel_end_time is not None:
                result[nvtx_event] = (kernel_start_time, kernel_end_time)
    return result

def parse_nvtx_kernel_events(conn: sqlite3.Connection, strings: dict, traceEvents: list, event_prefix=None, color_scheme={}):
    pid_to_device = link_pid_with_devices(conn)
    nvtx_events_per_device = parse_nvtx_events(conn, event_prefix=event_prefix, color_scheme=color_scheme)
    cuda_api_events_per_device = parse_cuda_api_events(conn, strings)
    cuda_kernel_events_per_device = parse_cupti_kernel_events(conn, strings)
    nvtx_kernel_event_map = link_nvtx_events_to_kernel_events(strings, pid_to_device, nvtx_events_per_device, cuda_api_events_per_device, cuda_kernel_events_per_device)
    for nvtx_event, (kernel_start_time, kernel_end_time) in nvtx_kernel_event_map.items():
        event = {
            "name": nvtx_event["text"],
            "ph": "X", # Complete Event (Begin + End event)
            "cat": "nvtx-kernel",
            "ts": munge_time(kernel_start_time),
            "dur": munge_time(kernel_end_time - kernel_start_time),
            "tid": "NVTX Kernel Thread {}".format(nvtx_event["tid"]),
            "pid": "Device {}".format(pid_to_device[nvtx_event["pid"]]),
            "args": {
                # TODO: More
                },
            }
        traceEvents.append(event)

def nsys2json():
    conn = sqlite3.connect(args.filename)
    conn.row_factory = sqlite3.Row

    strings = {}
    for r in conn.execute("SELECT id, value FROM StringIds"):
        strings[r["id"]] = r["value"]

    traceEvents = []
    for activity in args.activity_type:
        if activity == "kernel":
            parse_cupti_kernel_events(conn, strings, traceEvents)
        elif activity == "nvtx-cpu":
            parse_nvtx_events(conn, traceEvents, event_prefix=args.nvtx_event_prefix, color_scheme=args.nvtx_color_scheme)
        elif activity == "nvtx-kernel":
            parse_nvtx_kernel_events(conn, strings, traceEvents, event_prefix=args.nvtx_event_prefix, color_scheme=args.nvtx_color_scheme)
        else:
            raise ValueError(f"Unknown activity type: {activity}")
    # make the timelines appear in pid and tid order
    traceEvents.sort(key=lambda x: (x["pid"], x["tid"]))

    with open(args.output, 'w') as f:
        json.dump(traceEvents, f)

if __name__ == "__main__":
    nsys2json()