#!/bin/bash

# Claude Code Hooks Testing Script
# Usage: ./.claude/test-hooks.sh [--simulate] [--verbose]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CLAUDE_SETTINGS="$HOME/.claude/settings.json"
TEST_DIR="test_hooks_$$"
VERBOSE=false
SIMULATE=false

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

log_verbose() {
    if [[ "$VERBOSE" == true ]]; then
        echo -e "${NC}[VERBOSE]${NC} $1"
    fi
}

check_hooks_installed() {
    log_info "Checking hooks installation..."
    
    if [[ ! -f "$CLAUDE_SETTINGS" ]]; then
        log_error "Claude settings file not found: $CLAUDE_SETTINGS"
        log_error "Please run .claude/install-hooks.sh first"
        exit 1
    fi
    
    # Check if hooks are enabled
    if ! jq -e '.hookSettings.enableHooks == true' "$CLAUDE_SETTINGS" >/dev/null 2>&1; then
        log_warning "Hooks are not enabled in settings"
        return 1
    fi
    
    log_success "Hooks installation verified"
    return 0
}

test_json_validity() {
    log_info "Testing JSON configuration validity..."
    
    if jq empty "$CLAUDE_SETTINGS" 2>/dev/null; then
        log_success "Settings JSON is valid"
    else
        log_error "Settings JSON is invalid"
        return 1
    fi
    
    # Check required sections
    local sections=("hooks" "hookSettings")
    for section in "${sections[@]}"; do
        if jq -e "has(\"$section\")" "$CLAUDE_SETTINGS" >/dev/null; then
            log_verbose "Section '$section' found"
        else
            log_warning "Section '$section' missing"
        fi
    done
    
    return 0
}

test_tool_availability() {
    log_info "Testing tool availability..."
    
    local tools=(
        "uv:UV Python package manager"
        "git:Git version control"
        "jq:JSON processor"
    )
    
    local missing_tools=()
    
    for tool_desc in "${tools[@]}"; do
        IFS=':' read -r tool desc <<< "$tool_desc"
        
        if command -v "$tool" >/dev/null 2>&1; then
            log_success "$desc found"
            log_verbose "$tool version: $(${tool} --version 2>&1 | head -1)"
        else
            log_error "$desc not found"
            missing_tools+=("$tool")
        fi
    done
    
    # Test UV-based tools
    local uv_tools=(
        "ruff:Python linter"
        "pyright:Type checker"
        "pytest:Test runner"
    )
    
    for tool_desc in "${uv_tools[@]}"; do
        IFS=':' read -r tool desc <<< "$tool_desc"
        
        if uv run "$tool" --version >/dev/null 2>&1; then
            log_success "$desc available via UV"
            log_verbose "$tool version: $(uv run ${tool} --version 2>&1 | head -1)"
        else
            log_warning "$desc not available via UV"
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing critical tools: ${missing_tools[*]}"
        return 1
    fi
    
    return 0
}

simulate_hook_execution() {
    log_info "Simulating hook execution scenarios..."
    
    # Create test directory
    mkdir -p "$TEST_DIR"
    cd "$TEST_DIR"
    
    # Test Python file hook simulation
    log_info "Simulating Python file edit hooks..."
    
    local test_py="test_file.py"
    cat > "$test_py" << 'EOF'
def test_function():
    print("Hello World")
    return True

# Intentional style issue for ruff
def  bad_spacing():
    pass
EOF
    
    # Simulate ruff check
    log_verbose "Simulating: ruff check --fix $test_py"
    if [[ "$SIMULATE" == false ]]; then
        if uv run ruff check --fix "$test_py" 2>/dev/null; then
            log_success "Ruff simulation passed"
        else
            log_warning "Ruff simulation failed"
        fi
    else
        log_info "Ruff check simulation (skipped)"
    fi
    
    # Simulate pyright check
    log_verbose "Simulating: pyright $test_py"
    if [[ "$SIMULATE" == false ]]; then
        if uv run pyright "$test_py" >/dev/null 2>&1; then
            log_success "Pyright simulation passed"
        else
            log_warning "Pyright simulation failed"
        fi
    else
        log_info "Pyright check simulation (skipped)"
    fi
    
    # Test test file hooks
    local test_test="test_test.py"
    cat > "$test_test" << 'EOF'
import pytest

def test_simple():
    assert True

def test_failing():
    assert 1 == 1
EOF
    
    log_verbose "Simulating: pytest $test_test"
    if [[ "$SIMULATE" == false ]]; then
        if uv run pytest "$test_test" -v >/dev/null 2>&1; then
            log_success "Pytest simulation passed"
        else
            log_warning "Pytest simulation failed"
        fi
    else
        log_info "Pytest simulation (skipped)"
    fi
    
    # Cleanup
    cd ..
    rm -rf "$TEST_DIR"
    
    log_success "Hook simulation completed"
}

test_hook_patterns() {
    log_info "Testing hook pattern matching..."
    
    # Extract hook patterns from settings
    local patterns
    patterns=$(jq -r '.hooks.PostToolUse[].matcher | select(.file_paths) | .file_paths[]' "$CLAUDE_SETTINGS" 2>/dev/null || echo "")
    
    if [[ -n "$patterns" ]]; then
        log_success "Found file patterns in hooks:"
        while IFS= read -r pattern; do
            log_verbose "  - $pattern"
        done <<< "$patterns"
    else
        log_warning "No file patterns found in hooks"
    fi
    
    # Check tool name patterns
    local tool_patterns
    tool_patterns=$(jq -r '.hooks.PostToolUse[].matcher.tool_name' "$CLAUDE_SETTINGS" 2>/dev/null || echo "")
    
    if [[ -n "$tool_patterns" ]]; then
        log_success "Found tool name patterns:"
        while IFS= read -r pattern; do
            log_verbose "  - $pattern"
        done <<< "$tool_patterns"
    else
        log_warning "No tool name patterns found"
    fi
    
    return 0
}

test_environment_variables() {
    log_info "Testing hook environment variables..."
    
    # Test variables that would be available in real hooks
    local test_vars=(
        "HOME:$HOME"
        "PWD:$(pwd)"
        "PATH:$PATH"
    )
    
    for var_desc in "${test_vars[@]}"; do
        IFS=':' read -r var_name var_value <<< "$var_desc"
        
        if [[ -n "$var_value" ]]; then
            log_success "$var_name is set"
            log_verbose "$var_name = ${var_value:0:50}..."
        else
            log_warning "$var_name is not set"
        fi
    done
    
    # Test Claude-specific variables (simulated)
    log_info "Claude hook variables (simulated):"
    log_verbose "CLAUDE_FILE_PATHS = test1.py test2.py"
    log_verbose "CLAUDE_COMMAND = ruff check test.py"
    log_verbose "CLAUDE_NOTIFICATION = File modified"
    
    return 0
}

generate_test_report() {
    log_info "Generating test report..."
    
    local report_file="hooks_test_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
Claude Code Hooks Test Report
============================

Generated: $(date)
Test Directory: $(pwd)
Settings File: $CLAUDE_SETTINGS

Test Results:
EOF
    
    # Add test results (simplified)
    if check_hooks_installed >/dev/null 2>&1; then
        echo "✅ Hooks Installation: PASS" >> "$report_file"
    else
        echo "❌ Hooks Installation: FAIL" >> "$report_file"
    fi
    
    if test_json_validity >/dev/null 2>&1; then
        echo "✅ JSON Validity: PASS" >> "$report_file"
    else
        echo "❌ JSON Validity: FAIL" >> "$report_file"
    fi
    
    if test_tool_availability >/dev/null 2>&1; then
        echo "✅ Tool Availability: PASS" >> "$report_file"
    else
        echo "❌ Tool Availability: FAIL" >> "$report_file"
    fi
    
    cat >> "$report_file" << EOF

Hook Configuration Summary:
$(jq -r '.hooks | keys[]' "$CLAUDE_SETTINGS" 2>/dev/null | sed 's/^/- /')

Settings:
$(jq '.hookSettings' "$CLAUDE_SETTINGS" 2>/dev/null || echo "Not found")

EOF
    
    log_success "Test report generated: $report_file"
}

show_usage() {
    cat << EOF
Claude Code Hooks Testing Script

Usage: $0 [OPTIONS]

OPTIONS:
    --simulate  Simulate hook execution without running commands
    --verbose   Enable verbose output
    --help      Show this help message

EXAMPLES:
    $0                      # Full test with command execution
    $0 --simulate           # Test without executing commands
    $0 --verbose --simulate # Verbose simulation mode

EOF
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --simulate)
                SIMULATE=true
                shift
                ;;
            --verbose)
                VERBOSE=true
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
    
    # Main testing flow
    log_info "Starting Claude Code hooks testing..."
    echo "Simulation mode: $SIMULATE"
    echo "Verbose mode: $VERBOSE"
    echo
    
    local exit_code=0
    
    # Run tests
    check_hooks_installed || exit_code=1
    test_json_validity || exit_code=1
    test_tool_availability || exit_code=1
    test_hook_patterns || exit_code=1
    test_environment_variables || exit_code=1
    simulate_hook_execution || exit_code=1
    
    # Generate report
    generate_test_report
    
    echo
    if [[ $exit_code -eq 0 ]]; then
        log_success "All hooks tests completed successfully!"
    else
        log_warning "Some tests failed - check output above"
    fi
    
    echo
    log_info "To test hooks in real usage:"
    echo "  1. Edit a Python file: echo '# test' >> src/test.py"
    echo "  2. Run a bash command: ls -la"
    echo "  3. Check activity log: tail ~/.claude_activity.log"
    echo
    
    exit $exit_code
}

# Run main function
main "$@"