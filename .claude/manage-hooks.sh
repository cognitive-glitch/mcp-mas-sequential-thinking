#!/bin/bash

# Claude Code Hooks Management Script
# Usage: ./.claude/manage-hooks.sh [enable|disable|status|reset] [--quiet]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CLAUDE_SETTINGS="$HOME/.claude/settings.json"
PROJECT_HOOKS=".claude/settings.json"
QUIET=false

# Helper functions
log_info() {
    [[ "$QUIET" == false ]] && echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    [[ "$QUIET" == false ]] && echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    [[ "$QUIET" == false ]] && echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

check_settings_exist() {
    if [[ ! -f "$CLAUDE_SETTINGS" ]]; then
        log_error "Claude settings file not found: $CLAUDE_SETTINGS"
        log_error "Please run ./.claude/install-hooks.sh first"
        exit 1
    fi
}

check_jq_available() {
    if ! command -v jq >/dev/null 2>&1; then
        log_error "jq is required but not installed"
        log_error "Install with: apt install jq (Ubuntu) or brew install jq (macOS)"
        exit 1
    fi
}

get_hooks_status() {
    check_settings_exist
    check_jq_available
    
    local enabled
    enabled=$(jq -r '.hookSettings.enableHooks // false' "$CLAUDE_SETTINGS" 2>/dev/null)
    
    if [[ "$enabled" == "true" ]]; then
        echo "enabled"
    else
        echo "disabled"
    fi
}

enable_hooks() {
    log_info "Enabling Claude Code hooks..."
    
    check_settings_exist
    check_jq_available
    
    # Update enableHooks to true
    local temp_file
    temp_file=$(mktemp)
    
    if jq '.hookSettings.enableHooks = true' "$CLAUDE_SETTINGS" > "$temp_file"; then
        mv "$temp_file" "$CLAUDE_SETTINGS"
        log_success "Hooks enabled successfully"
    else
        rm -f "$temp_file"
        log_error "Failed to enable hooks"
        exit 1
    fi
}

disable_hooks() {
    log_info "Disabling Claude Code hooks..."
    
    check_settings_exist
    check_jq_available
    
    # Update enableHooks to false
    local temp_file
    temp_file=$(mktemp)
    
    if jq '.hookSettings.enableHooks = false' "$CLAUDE_SETTINGS" > "$temp_file"; then
        mv "$temp_file" "$CLAUDE_SETTINGS"
        log_success "Hooks disabled successfully"
    else
        rm -f "$temp_file"
        log_error "Failed to disable hooks"
        exit 1
    fi
}

show_status() {
    check_settings_exist
    check_jq_available
    
    local status
    status=$(get_hooks_status)
    
    echo
    echo "Claude Code Hooks Status"
    echo "========================"
    echo
    echo "Settings file: $CLAUDE_SETTINGS"
    echo "Status: $status"
    echo
    
    if [[ "$status" == "enabled" ]]; then
        echo -e "${GREEN}✅ Hooks are ENABLED${NC}"
        
        # Show hook categories
        local categories
        categories=$(jq -r '.hooks | keys[]' "$CLAUDE_SETTINGS" 2>/dev/null || echo "")
        
        if [[ -n "$categories" ]]; then
            echo
            echo "Active hook categories:"
            while IFS= read -r category; do
                local count
                count=$(jq -r ".hooks.${category} | length" "$CLAUDE_SETTINGS" 2>/dev/null || echo "0")
                echo "  • $category ($count hooks)"
            done <<< "$categories"
        fi
        
        # Show settings
        echo
        echo "Hook settings:"
        jq -r '.hookSettings | to_entries[] | "  • \(.key): \(.value)"' "$CLAUDE_SETTINGS" 2>/dev/null || echo "  • No settings found"
        
    else
        echo -e "${RED}❌ Hooks are DISABLED${NC}"
    fi
    
    echo
    echo "Management commands:"
    echo "  Enable:  ./.claude/manage-hooks.sh enable"
    echo "  Disable: ./.claude/manage-hooks.sh disable"
    echo "  Reset:   ./.claude/manage-hooks.sh reset"
    echo
}

reset_hooks() {
    log_info "Resetting hooks to project defaults..."
    
    if [[ ! -f "$PROJECT_HOOKS" ]]; then
        log_error "Project hooks file not found: $PROJECT_HOOKS"
        log_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Create backup
    local backup_file="${CLAUDE_SETTINGS}.backup.$(date +%Y%m%d_%H%M%S)"
    if [[ -f "$CLAUDE_SETTINGS" ]]; then
        log_info "Creating backup: $backup_file"
        cp "$CLAUDE_SETTINGS" "$backup_file"
    fi
    
    # Update project-specific paths and copy
    local project_path
    project_path="$(pwd)"
    
    local temp_settings
    temp_settings=$(mktemp)
    sed "s|/home/dev/GitHub/reflective-sequential-thinking-mcp|$project_path|g" "$PROJECT_HOOKS" > "$temp_settings"
    
    # Create Claude directory if needed
    mkdir -p "$(dirname "$CLAUDE_SETTINGS")"
    
    # Copy updated settings
    mv "$temp_settings" "$CLAUDE_SETTINGS"
    
    log_success "Hooks reset to project defaults"
    log_success "Previous settings backed up to: $backup_file"
}

list_hook_files() {
    log_info "Scanning for hook-related files..."
    
    echo
    echo "Hook Files Found:"
    echo "=================="
    
    # Check project hooks
    if [[ -f "$PROJECT_HOOKS" ]]; then
        echo "✅ Project hooks: $PROJECT_HOOKS"
    else
        echo "❌ Project hooks: $PROJECT_HOOKS (missing)"
    fi
    
    # Check installed hooks
    if [[ -f "$CLAUDE_SETTINGS" ]]; then
        echo "✅ Installed hooks: $CLAUDE_SETTINGS"
        local file_size
        file_size=$(stat -c%s "$CLAUDE_SETTINGS" 2>/dev/null || stat -f%z "$CLAUDE_SETTINGS" 2>/dev/null || echo "unknown")
        echo "   Size: $file_size bytes"
        local mod_time
        mod_time=$(stat -c%y "$CLAUDE_SETTINGS" 2>/dev/null || stat -f%Sm "$CLAUDE_SETTINGS" 2>/dev/null || echo "unknown")
        echo "   Modified: $mod_time"
    else
        echo "❌ Installed hooks: $CLAUDE_SETTINGS (missing)"
    fi
    
    # Check for backups
    local backup_pattern="${CLAUDE_SETTINGS}.backup.*"
    local backups
    backups=$(ls $backup_pattern 2>/dev/null || echo "")
    
    if [[ -n "$backups" ]]; then
        echo
        echo "Backup files found:"
        while IFS= read -r backup; do
            echo "  📁 $backup"
        done <<< "$backups"
    else
        echo
        echo "No backup files found"
    fi
    
    # Check activity log
    local activity_log="$HOME/.claude_activity.log"
    if [[ -f "$activity_log" ]]; then
        echo
        echo "✅ Activity log: $activity_log"
        local log_lines
        log_lines=$(wc -l < "$activity_log" 2>/dev/null || echo "0")
        echo "   Lines: $log_lines"
    else
        echo
        echo "❌ Activity log: $activity_log (not created yet)"
    fi
    
    echo
}

cleanup_hooks() {
    log_info "Cleaning up hook-related files..."
    
    echo -n "This will remove all Claude hooks and backups. Continue? [y/N]: "
    read -r response
    
    if [[ ! "$response" =~ ^[yY]$ ]]; then
        log_info "Cleanup cancelled"
        exit 0
    fi
    
    local removed_files=()
    
    # Remove main settings file
    if [[ -f "$CLAUDE_SETTINGS" ]]; then
        rm "$CLAUDE_SETTINGS"
        removed_files+=("$CLAUDE_SETTINGS")
    fi
    
    # Remove backup files
    local backup_pattern="${CLAUDE_SETTINGS}.backup.*"
    local backups
    backups=$(ls $backup_pattern 2>/dev/null || echo "")
    
    if [[ -n "$backups" ]]; then
        while IFS= read -r backup; do
            rm "$backup"
            removed_files+=("$backup")
        done <<< "$backups"
    fi
    
    # Optionally remove activity log
    local activity_log="$HOME/.claude_activity.log"
    if [[ -f "$activity_log" ]]; then
        echo -n "Remove activity log? [y/N]: "
        read -r response
        
        if [[ "$response" =~ ^[yY]$ ]]; then
            rm "$activity_log"
            removed_files+=("$activity_log")
        fi
    fi
    
    if [[ ${#removed_files[@]} -gt 0 ]]; then
        log_success "Removed ${#removed_files[@]} files:"
        for file in "${removed_files[@]}"; do
            echo "  🗑️  $file"
        done
    else
        log_info "No files to remove"
    fi
}

show_usage() {
    cat << EOF
Claude Code Hooks Management Script

Usage: $0 COMMAND [OPTIONS]

COMMANDS:
    enable      Enable Claude Code hooks
    disable     Disable Claude Code hooks  
    status      Show current hooks status
    reset       Reset hooks to project defaults
    list        List all hook-related files
    cleanup     Remove all hooks and backups
    help        Show this help message

OPTIONS:
    --quiet     Suppress informational output

EXAMPLES:
    $0 status                # Show current status
    $0 enable                # Enable hooks
    $0 disable --quiet       # Disable hooks quietly
    $0 reset                 # Reset to project defaults

EOF
}

main() {
    local command=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            enable|disable|status|reset|list|cleanup|help)
                command="$1"
                shift
                ;;
            --quiet)
                QUIET=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Execute command
    case "$command" in
        enable)
            enable_hooks
            ;;
        disable)
            disable_hooks
            ;;
        status)
            show_status
            ;;
        reset)
            reset_hooks
            ;;
        list)
            list_hook_files
            ;;
        cleanup)
            cleanup_hooks
            ;;
        help|"")
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"