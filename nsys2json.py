import sqlite3
import argparse
import json
import re

# Code adapted from https://github.com/ezyang/nvprof2json

parser = argparse.ArgumentParser(description='Convert nsight systems sqlite output to Google Event Trace compatible JSON.')
parser.add_argument('--filename')
parser.add_argument("-o", "--output", help="Output file name", required=True)
parser.add_argument("-t", "--activity-type", help="Type of activities shown. Default to all.", default=["kernel", "nvtx"], choices=['kernel', 'nvtx'], nargs="+")
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

def parse_cupti_kernel_events(conn: sqlite3.Connection, strings: dict, traceEvents: list):
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
    for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"):
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

def parse_nvtx_events(conn: sqlite3.Connection, traceEvents: list, event_prefix=None, color_scheme={}):
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
    # map each pid to a device. assumes each pid is associated with a single device
    pid_to_device = {}
    for row in conn.execute("SELECT DISTINCT deviceId, globalPid / 0x1000000 % 0x1000000 AS PID FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        assert row["PID"] not in pid_to_device, \
            f"A single PID ({row['PID']}) is associated with multiple devices ({pid_to_device[row['PID']]} and {row['deviceId']})."
        pid_to_device[row["PID"]] = row["deviceId"]

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

    # eventType 59 is NvtxPushPopRange, which corresponds to torch.cuda.nvtx.range apis
    for row in conn.execute(f"SELECT start, end, text, globalTid / 0x1000000 % 0x1000000 AS PID, globalTid % 0x1000000 AS TID FROM NVTX_EVENTS WHERE NVTX_EVENTS.eventType == 59{match_text};"):
        text = row['text']
        pid = row['PID']
        tid = row['TID']
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
        elif activity == "nvtx":
            parse_nvtx_events(conn, traceEvents, event_prefix=args.nvtx_event_prefix, color_scheme=args.nvtx_color_scheme)
        else:
            raise ValueError(f"Unknown activity type: {activity}")
    # make the timelines appear in pid and tid order
    traceEvents.sort(key=lambda x: (x["pid"], x["tid"]))

    with open(args.output, 'w') as f:
        json.dump(traceEvents, f)

if __name__ == "__main__":
    nsys2json()