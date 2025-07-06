#!/bin/bash

# Claude Code Hooks Installation Script
# Usage: ./.claude/install-hooks.sh [--force] [--backup] [--test]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CLAUDE_DIR="$HOME/.claude"
SETTINGS_FILE="$CLAUDE_DIR/settings.json"
PROJECT_HOOKS=".claude/settings.json"
BACKUP_SUFFIX=".backup.$(date +%Y%m%d_%H%M%S)"

# Flags
FORCE_INSTALL=false
CREATE_BACKUP=true
TEST_MODE=false

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if running in project directory
    if [[ ! -f "$PROJECT_HOOKS" ]]; then
        log_error "Project hooks file not found: $PROJECT_HOOKS"
        log_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Check required tools
    local missing_tools=()
    
    command -v uv >/dev/null 2>&1 || missing_tools+=("uv")
    command -v git >/dev/null 2>&1 || missing_tools+=("git")
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install missing tools before proceeding"
        exit 1
    fi
    
    # Check Python tools via uv
    if ! uv run ruff --version >/dev/null 2>&1; then
        log_warning "ruff not available via uv - some hooks may fail"
    fi
    
    if ! uv run pyright --version >/dev/null 2>&1; then
        log_warning "pyright not available via uv - some hooks may fail"
    fi
    
    if ! uv run pytest --version >/dev/null 2>&1; then
        log_warning "pytest not available via uv - test hooks may fail"
    fi
    
    log_success "Prerequisites check completed"
}

create_claude_directory() {
    if [[ ! -d "$CLAUDE_DIR" ]]; then
        log_info "Creating Claude directory: $CLAUDE_DIR"
        mkdir -p "$CLAUDE_DIR"
    fi
}

backup_existing_settings() {
    if [[ -f "$SETTINGS_FILE" && "$CREATE_BACKUP" == true ]]; then
        local backup_file="${SETTINGS_FILE}${BACKUP_SUFFIX}"
        log_info "Backing up existing settings to: $backup_file"
        cp "$SETTINGS_FILE" "$backup_file"
        log_success "Backup created"
    fi
}

install_hooks() {
    log_info "Installing Claude Code hooks..."
    
    if [[ -f "$SETTINGS_FILE" && "$FORCE_INSTALL" != true ]]; then
        log_warning "Existing settings file found: $SETTINGS_FILE"
        echo -n "Overwrite existing settings? [y/N]: "
        read -r response
        
        if [[ ! "$response" =~ ^[yY]$ ]]; then
            log_info "Installation cancelled by user"
            exit 0
        fi
    fi
    
    # Update project-specific paths in hooks configuration
    local project_path
    project_path="$(pwd)"
    
    # Create temporary settings with updated paths
    local temp_settings="/tmp/claude_settings_$$.json"
    sed "s|/home/dev/GitHub/reflective-sequential-thinking-mcp|$project_path|g" "$PROJECT_HOOKS" > "$temp_settings"
    
    # Copy updated settings
    cp "$temp_settings" "$SETTINGS_FILE"
    rm "$temp_settings"
    
    log_success "Hooks installed successfully"
}

test_hooks() {
    log_info "Testing hooks configuration..."
    
    # Test JSON validity
    if ! jq empty "$SETTINGS_FILE" 2>/dev/null; then
        log_error "Invalid JSON in settings file"
        return 1
    fi
    
    # Test specific tools
    local test_file="test_hooks_$$.py"
    echo "# Test file for hooks" > "$test_file"
    
    log_info "Testing ruff..."
    if uv run ruff check "$test_file" >/dev/null 2>&1; then
        log_success "ruff test passed"
    else
        log_warning "ruff test failed"
    fi
    
    log_info "Testing pyright..."
    if uv run pyright "$test_file" >/dev/null 2>&1; then
        log_success "pyright test passed"
    else
        log_warning "pyright test failed"
    fi
    
    # Cleanup
    rm -f "$test_file"
    
    log_success "Hooks testing completed"
}

show_usage() {
    cat << EOF
Claude Code Hooks Installation Script

Usage: $0 [OPTIONS]

OPTIONS:
    --force     Force installation without confirmation
    --no-backup Don't backup existing settings  
    --test      Test hooks after installation
    --help      Show this help message

EXAMPLES:
    $0                    # Interactive installation with backup
    $0 --force --test     # Force install and test
    $0 --no-backup        # Install without backup

EOF
}

show_status() {
    log_info "Installation Status:"
    echo "  Project hooks: $PROJECT_HOOKS"
    echo "  Claude directory: $CLAUDE_DIR"
    echo "  Settings file: $SETTINGS_FILE"
    echo "  Force install: $FORCE_INSTALL"
    echo "  Create backup: $CREATE_BACKUP"
    echo "  Test mode: $TEST_MODE"
    echo
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force)
                FORCE_INSTALL=true
                shift
                ;;
            --no-backup)
                CREATE_BACKUP=false
                shift
                ;;
            --test)
                TEST_MODE=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Main installation flow
    log_info "Starting Claude Code hooks installation..."
    show_status
    
    check_prerequisites
    create_claude_directory
    backup_existing_settings
    install_hooks
    
    if [[ "$TEST_MODE" == true ]]; then
        test_hooks
    fi
    
    # Show completion message
    echo
    log_success "Claude Code hooks installation completed!"
    echo
    echo "Next steps:"
    echo "  1. Edit a Python file to test PostToolUse hooks"
    echo "  2. Run a bash command to test PreToolUse hooks"
    echo "  3. Check ~/.claude_activity.log for hook activity"
    echo
    echo "To disable hooks temporarily:"
    echo "  mv ~/.claude/settings.json ~/.claude/settings.json.disabled"
    echo
    echo "To re-enable hooks:"
    echo "  mv ~/.claude/settings.json.disabled ~/.claude/settings.json"
    echo
}

# Run main function
main "$@"