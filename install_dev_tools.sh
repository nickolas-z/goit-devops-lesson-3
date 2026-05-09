#!/usr/bin/env bash
set -euo pipefail

# Color codes for logging
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

LOG_FILE="$(pwd)/install.log"

###########################################################################################
# Logging function with timestamp and color-coded levels
log() {
  local level="$1"; shift
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*"
  echo "$msg" >> "$LOG_FILE"
  case "$level" in
    INFO)  echo -e "${GREEN}${msg}${NC}" ;;
    WARN)  echo -e "${YELLOW}${msg}${NC}" ;;
    ERROR) echo -e "${RED}${msg}${NC}" ;;
    *)     echo "$msg" ;;
  esac
}

###########################################################################################
# Helpers to check for command availability and versions
check_cmd() {
  command -v "$1" &>/dev/null
}

check_docker_compose() {
  docker compose version &>/dev/null 2>&1 || check_cmd docker-compose
}

docker_compose_version() {
  if docker compose version &>/dev/null 2>&1; then
    docker compose version
  elif check_cmd docker-compose; then
    docker-compose --version
  else
    return 1
  fi
}

check_python_pip() {
  python3 -m pip --version &>/dev/null 2>&1
}

in_virtualenv() {
  python3 -c 'import sys; raise SystemExit(0 if sys.prefix != sys.base_prefix else 1)' \
    &>/dev/null
}

pip_supports_break_system_packages() {
  python3 -m pip install --help 2>/dev/null | grep -q -- '--break-system-packages'
}

python_is_externally_managed() {
  python3 -c 'import pathlib, sysconfig; p = pathlib.Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED"; raise SystemExit(0 if p.exists() else 1)' \
    &>/dev/null
}

# Returns 0 if python3 version >= 3.9, 1 otherwise
check_python_version() {
  if ! check_cmd python3; then
    return 1
  fi
  local ver
  ver="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  local major minor
  major="$(echo "$ver" | cut -d. -f1)"
  minor="$(echo "$ver" | cut -d. -f2)"
  if [[ "$major" -gt 3 ]] || { [[ "$major" -eq 3 ]] && [[ "$minor" -ge 9 ]]; }; then
    return 0
  fi
  return 1
}

###########################################################################################
# Docker installation
install_docker() {
  log "INFO" "Checking Docker installation..."

  if check_cmd docker; then
    log "INFO" "Docker already installed: $(docker --version)"
  else
    log "INFO" "Installing Docker via official Docker Inc. apt repository..."

    # Prerequisites
    apt-get update -qq
    apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg lsb-release

    # GPG key
    install -m 0755 -d /etc/apt/keyrings
    # Detect distro for the correct Docker apt repository (ubuntu or debian).
    local distro_id
    distro_id="$(. /etc/os-release 2>/dev/null && echo "${ID:-ubuntu}")"
    if [[ "$distro_id" != "ubuntu" && "$distro_id" != "debian" ]]; then
      log "WARN" "Distro '$distro_id' is not ubuntu/debian — falling back to ubuntu Docker repo."
      distro_id="ubuntu"
    fi
    curl -fsSL "https://download.docker.com/linux/${distro_id}/gpg" \
      | gpg --batch --yes --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    # Repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/${distro_id} \
$(lsb_release -cs) stable" \
      | tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update -qq
    apt-get install -y --no-install-recommends \
      docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    log "INFO" "Docker installed: $(docker --version)"
  fi

  # Docker Compose v2 plugin or legacy docker-compose binary.
  log "INFO" "Checking Docker Compose installation..."
  if check_docker_compose; then
    log "INFO" "Docker Compose already available: $(docker_compose_version)"
  else
    log "WARN" "Docker Compose v2 plugin not found. Installing..."
    apt-get install -y --no-install-recommends docker-compose-plugin
    log "INFO" "Docker Compose installed: $(docker_compose_version)"
  fi
}

###########################################################################################
# Python installation
install_python() {
  log "INFO" "Checking Python 3.9+ installation..."

  if check_python_version; then
    log "INFO" "Python already satisfies >= 3.9: $(python3 --version)"
  else
    log "INFO" "System Python < 3.9 or missing. Installing python3.9 via apt..."
    log "INFO" "Refreshing apt package metadata before checking python3.9 availability..."
    apt-get update -qq

    # Try deadsnakes PPA for newer Ubuntu; fallback to plain apt
    if apt-cache show python3.9 &>/dev/null 2>&1; then
      apt-get install -y --no-install-recommends python3.9 python3.9-venv python3.9-dev
      # Make python3 point to 3.9 if not already
      update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 || true
    else
      log "WARN" "python3.9 not in apt cache; trying deadsnakes PPA..."
      apt-get install -y --no-install-recommends software-properties-common
      add-apt-repository -y ppa:deadsnakes/ppa
      apt-get update -qq
      apt-get install -y --no-install-recommends python3.9 python3.9-venv python3.9-dev
      update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 || true
    fi

    log "INFO" "Python installed: $(python3 --version)"
  fi
}

###########################################################################################
# pip installation
install_pip() {
  log "INFO" "Checking pip installation for python3..."

  if check_python_pip; then
    log "INFO" "pip already installed: $(python3 -m pip --version)"
  else
    log "INFO" "Installing python3-pip via apt..."
    apt-get install -y --no-install-recommends python3-pip
    if ! check_python_pip; then
      log "ERROR" "python3-pip was installed, but 'python3 -m pip' is still unavailable."
      return 1
    fi
    log "INFO" "pip installed: $(python3 -m pip --version)"
  fi
}

###########################################################################################
# Configure extra pip install options for root/non-root and PEP 668 systems.
PIP_INSTALL_ARGS=()

configure_pip_install_args() {
  PIP_INSTALL_ARGS=()

  if in_virtualenv; then
    log "INFO" "Active Python virtualenv detected; pip packages will be installed into it."
    return 0
  fi

  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    PIP_INSTALL_ARGS+=(--user)
    log "INFO" "Non-root shell outside venv; missing pip packages will be installed with --user."
  else
    log "INFO" "Root shell outside venv; missing pip packages will be installed for system Python."
  fi

  if python_is_externally_managed; then
    if pip_supports_break_system_packages; then
      PIP_INSTALL_ARGS+=(--break-system-packages)
      log "WARN" "Externally managed Python detected; adding --break-system-packages for pip installs."
    else
      log "WARN" "Externally managed Python detected, but pip does not support --break-system-packages."
    fi
  fi
}

###########################################################################################
# pip packages
install_pip_packages() {
  log "INFO" "Checking pip packages (torch, torchvision, pillow, django)..."

  log "INFO" "Using pip: $(python3 -m pip --version)"

  local -a missing=()

  for pkg in torch torchvision pillow django; do
    if python3 -m pip show "$pkg" &>/dev/null 2>&1; then
      local ver
      ver="$(python3 -m pip show "$pkg" 2>/dev/null | awk '/^Version:/ {print $2}')"
      log "INFO" "${pkg} already installed: ${ver}"
    else
      missing+=("$pkg")
    fi
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    log "INFO" "All pip packages already installed — nothing to do."
    return 0
  fi

  local -a torch_missing=()
  local -a pypi_missing=()
  for pkg in "${missing[@]}"; do
    case "$pkg" in
      torch|torchvision) torch_missing+=("$pkg") ;;
      *) pypi_missing+=("$pkg") ;;
    esac
  done

  if [[ ${#torch_missing[@]} -gt 0 ]]; then
    log "INFO" "Installing CPU-only PyTorch packages: ${torch_missing[*]}"
    python3 -m pip install \
      "${PIP_INSTALL_ARGS[@]}" \
      "${torch_missing[@]}" \
      --index-url https://download.pytorch.org/whl/cpu \
      2>&1 | tee -a "$LOG_FILE"
  fi

  if [[ ${#pypi_missing[@]} -gt 0 ]]; then
    log "INFO" "Installing PyPI packages: ${pypi_missing[*]}"
    python3 -m pip install \
      "${PIP_INSTALL_ARGS[@]}" \
      "${pypi_missing[@]}" \
      2>&1 | tee -a "$LOG_FILE"
  fi

  log "INFO" "All pip packages installed."
}

###########################################################################################
# Version report
print_versions() {
  log "INFO" "Tool versions:"

  if check_cmd docker; then
    log "INFO" "docker       : $(docker --version)"
  else
    log "WARN" "docker       : NOT FOUND"
  fi

  if check_docker_compose; then
    log "INFO" "docker compose: $(docker_compose_version)"
  else
    log "WARN" "docker compose: NOT FOUND"
  fi

  if check_cmd python3; then
    log "INFO" "python3      : $(python3 --version)"
  else
    log "WARN" "python3      : NOT FOUND"
  fi

  if check_python_pip; then
    log "INFO" "pip          : $(python3 -m pip --version)"
  else
    log "WARN" "pip          : NOT FOUND for python3"
  fi

  # Python packages via python3 -m pip show
  for pkg in torch torchvision pillow django; do
    local ver
    ver="$(python3 -m pip show "$pkg" 2>/dev/null | awk '/^Version:/ {print $2}')" || true
    if [[ -n "$ver" ]]; then
      log "INFO" "${pkg}         : ${ver}"
    else
      log "WARN" "${pkg}         : NOT INSTALLED"
    fi
  done
}

###########################################################################################
# Entry point
main() {
  log "INFO" "Starting environment setup..."

  # Root check: system-level installs (Docker, Python, pip) require root.
  # If not root, verify all system tools are already present before proceeding.
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    local needs_root=false
    check_cmd docker                   || needs_root=true
    check_docker_compose               || needs_root=true
    check_python_version               || needs_root=true
    check_python_pip                   || needs_root=true

    if [[ "$needs_root" == true ]]; then
      log "ERROR" "Root privileges required: Docker, Docker Compose, Python, or pip are missing."
      log "ERROR" "Re-run with: sudo $0"
      exit 1
    fi

    log "WARN" "Not running as root, but all system tools are present."
    log "WARN" "Proceeding with non-privileged pip package checks/installation."
  fi

  install_docker
  install_python
  install_pip
  configure_pip_install_args
  install_pip_packages
  print_versions

  log "INFO" "Setup completed."
  exit 0
}

main "$@"
