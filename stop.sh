#!/bin/bash
# stop.sh — stop the lane: coordinator, then both Spark rank tenants (graceful MPS quit).
exec "$(dirname "$0")/start.sh" down
