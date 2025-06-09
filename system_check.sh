#!/bin/bash

# J.A.R.V.I.S Enhanced System Check v2.1
# Author: Enhanced by AI Assistant
# Version: 2.1

# Kleuren
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Config
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOGFILE="jarvis_system_check_$TIMESTAMP.log"
TEMP_DIR="/tmp/jarvis_check"
WARNING_CPU_THRESHOLD=80
WARNING_MEM_THRESHOLD=85
WARNING_DISK_THRESHOLD=90

# Setup logging
mkdir -p "$TEMP_DIR"
exec > >(tee -a "$LOGFILE") 2>&1

# Functies
print_header() {
    local title="$1"
    local icon="$2"
    echo -e "\n${CYAN}${icon} ======== ${title} ========${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}❌ ERROR: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

check_command() {
    command -v "$1" &>/dev/null
}

get_load_color() {
    local load=$1
    if (( $(echo "$load > 80" | bc -l) )); then
        echo -e "${RED}"
    elif (( $(echo "$load > 60" | bc -l) )); then
        echo -e "${YELLOW}"
    else
        echo -e "${GREEN}"
    fi
}

# LOGO
clear
echo -e "${PURPLE}"
cat << "EOF"
     ██╗   █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║  ██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║  ███████║██████╔╝██║   ██║██║███████╗
██   ██║  ██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝  ██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝   ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
EOF
echo -e "${NC}"

echo -e "${GREEN}🔧 System Check-up gestart op $(date)${NC}"
echo -e "${BLUE}💻 Hostname: $(hostname) | 👤 User: $(whoami)${NC}"

# SYSTEM INFO
print_header "SYSTEM INFORMATION" "🖥️"
[ -f /etc/os-release ] && source /etc/os-release && echo -e "OS: ${GREEN}$PRETTY_NAME${NC}"
echo -e "Kernel: ${GREEN}$(uname -r)${NC} | Arch: ${GREEN}$(uname -m)${NC}"

# CPU INFO
print_header "CPU INFORMATION" "🧠"
if check_command lscpu; then
    CPU_MODEL=$(lscpu | grep "Model name" | cut -d':' -f2 | xargs)
    CPU_CORES=$(lscpu | grep "^CPU(s):" | cut -d':' -f2 | xargs)
    CPU_THREADS=$(lscpu | grep "Thread(s) per core" | cut -d':' -f2 | xargs)
    echo -e "Model: ${GREEN}$CPU_MODEL${NC} | Cores: ${GREEN}$CPU_CORES${NC} | Threads/Core: ${GREEN}$CPU_THREADS${NC}"
    
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk -F'id,' -v prefix="${GREEN}" '{ split($1, vs, ","); print 100 - vs[length(vs)] }')
    CPU_COLOR=$(get_load_color "$CPU_USAGE")
    echo -e "Current Usage: ${CPU_COLOR}${CPU_USAGE}%${NC}"
    (( $(echo "$CPU_USAGE > $WARNING_CPU_THRESHOLD" | bc -l) )) && print_warning "High CPU usage detected!"
fi

# MEMORY
print_header "MEMORY INFORMATION" "📦"
if check_command free; then
    MEM_PERCENT=$(free | awk '/Mem:/ {printf("%.1f", $3/$2 * 100)}')
    MEM_COLOR=$(get_load_color "$MEM_PERCENT")
    echo -e "Memory Usage: ${MEM_COLOR}${MEM_PERCENT}%${NC}"
    free -h
    (( $(echo "$MEM_PERCENT > $WARNING_MEM_THRESHOLD" | bc -l) )) && print_warning "High memory usage detected!"
fi

# DISK
print_header "DISK USAGE" "💽"
if check_command df; then
    echo -e "${YELLOW}Filesystem breakdown:${NC}"
    df -h --output=source,fstype,size,used,avail,pcent,target | grep -vE 'tmpfs|udev'
    echo -e "\n${YELLOW}Total disk usage:${NC}"
    df -h --total | grep total
    df --output=pcent,target | tail -n +2 | grep -v tmpfs | awk -v t="$WARNING_DISK_THRESHOLD" '{gsub(/%/, "", $1); if ($1 > t) print_warning("High disk usage on " $2 " (" $1 "%)")}'
fi

# LOAD & UPTIME
print_header "SYSTEM LOAD & UPTIME" "⏱️"
check_command uptime && echo -e "Uptime: ${GREEN}$(uptime -p)${NC} | Load: ${GREEN}$(uptime | awk -F'load average: ' '{print $2}')${NC}"

# NETWORK
print_header "NETWORK INFORMATION" "🌐"
check_command hostname && echo -e "IP: ${GREEN}$(hostname -I)${NC}"
check_command ss && echo -e "Listening Ports: ${GREEN}$(ss -tuln | grep LISTEN | wc -l)${NC}"

# PROCESSES
print_header "TOP PROCESSES" "🔥"
echo -e "${YELLOW}Top 5 CPU-intensive processes:${NC}"
ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu | head -n 6
echo -e "${YELLOW}Top 5 Memory-intensive processes:${NC}"
ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | head -n 6

# SERVICES
if check_command systemctl; then
    print_header "CRITICAL SERVICES STATUS" "⚙️"
    for s in ssh networking systemd-resolved cron; do
        if systemctl list-units --full -all | grep -q "$s.service"; then
            systemctl is-active --quiet "$s" && print_success "$s is running" || print_error "$s is not running"
        fi
    done
fi

# UPDATES
print_header "PACKAGE UPDATES" "📦"
if check_command apt; then
    echo "Checking for updates..."
    sudo apt update -qq &>/dev/null
    UPGRADABLE=$(apt list --upgradable 2>/dev/null | grep -c upgradable)
    [ "$UPGRADABLE" -gt 1 ] && print_warning "$((UPGRADABLE-1)) packages can be upgraded" || print_success "System is up to date"
elif check_command yum; then
    UPDATES=$(yum check-update --quiet | wc -l)
    [ "$UPDATES" -gt 0 ] && print_warning "$UPDATES packages can be updated" || print_success "System is up to date"
fi

# SECURITY
print_header "SECURITY CHECK" "🔒"
if [ -f /var/log/auth.log ]; then
    FAILED=$(grep "Failed password" /var/log/auth.log | tail -n 10 | wc -l)
    [ "$FAILED" -gt 0 ] && print_warning "$FAILED recent failed login attempts" || print_success "No recent failed logins"
fi

# TEMP
if check_command sensors; then
    print_header "TEMPERATURE MONITORING" "🌡️"
    sensors | grep -E "(Core|temp)" | head -n 5
fi

# Updates Check
print_header "PACKAGE UPDATES" "📦"
if check_command "apt"; then
    echo "Checking and applying updates..."
    sudo apt update -qq && sudo apt upgrade -y
    if [ $? -eq 0 ]; then
        print_success "All available APT packages are updated"
    else
        print_error "Failed to update packages via APT"
    fi
elif check_command "yum"; then
    echo "Checking and applying updates..."
    sudo yum update -y
    if [ $? -eq 0 ]; then
        print_success "All available YUM packages are updated"
    else
        print_error "Failed to update packages via YUM"
    fi
elif check_command "dnf"; then
    echo "Checking and applying updates..."
    sudo dnf upgrade --refresh -y
    if [ $? -eq 0 ]; then
        print_success "All available DNF packages are updated"
    else
        print_error "Failed to update packages via DNF"
    fi
else
    print_warning "No supported package manager found"
fi


# SUMMARY
print_header "SUMMARY" "📊"
echo -e "Check voltooid op: ${GREEN}$(date)${NC}"
echo -e "Log opgeslagen als: ${GREEN}$LOGFILE${NC}"
echo -e "Temp folder: ${GREEN}$TEMP_DIR${NC}"

# Cleanup
rm -rf "$TEMP_DIR" 2>/dev/null

echo -e "\n${PURPLE}✅ Systeemscan afgerond. Blijf scherp, Stark.${NC}"
echo -e "${CYAN}💡 Tip: Run 'watch -n 30 ./jarvis_check.sh' voor realtime systeemmonitoring${NC}"
