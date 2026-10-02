#!/usr/bin/env bash
# Print headroom before launching jobs. Usage: tools/capacity.sh
free -m | awk '/Mem:/ {printf "mem: avail %d MB of %d MB\n", $7, $2}'
uptime | sed 's/.*load average/load average/'
echo "our python procs:"; ps -eo pid,rss,pcpu,args --sort=-rss | grep -E "[.]venv312/bin/python" | awk '{printf "  %s %5d MB %5s%% %s %s %s\n", $1, $2/1024, $3, $5, $6, $7}' | head -20
ps -eo rss,args | grep -E "[.]venv312/bin/python" | awk '{s+=$1} END {printf "total our RSS: %d MB\n", s/1024}'
