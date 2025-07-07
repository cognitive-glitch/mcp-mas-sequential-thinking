#!/bin/bash

# Comprehensive test runner for the Reflective Thinking MCP Server
# Runs all test suites and generates a summary report

echo "🧪 Running Comprehensive Test Suite for Reflective Thinking MCP Server"
echo "=================================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test categories
declare -A test_suites=(
    ["Unit Tests - Thought Models"]="tests/test_thought_models.py"
    ["Unit Tests - Shared Context"]="tests/test_shared_context.py"
    ["Unit Tests - Tool Selector"]="tests/test_tool_selector.py"
    ["Unit Tests - Providers"]="tests/test_providers.py"
    ["Unit Tests - Advanced Context"]="tests/test_context_advanced.py"
    ["Integration Tests - MCP Endpoints"]="tests/test_mcp_integration.py"
    ["Integration Tests - Error Handling"]="tests/test_error_handling.py"
    ["Integration Tests - Real Scenarios"]="tests/test_scenarios.py"
)

# Summary tracking
total_tests=0
passed_tests=0
failed_tests=0
skipped_tests=0
errors=0

echo -e "${BLUE}Running test suites...${NC}"
echo ""

# Run each test suite
for suite_name in "${!test_suites[@]}"; do
    test_file="${test_suites[$suite_name]}"
    
    echo -e "${YELLOW}Running: $suite_name${NC}"
    echo "File: $test_file"
    echo "----------------------------------------"
    
    # Run tests and capture output
    if [ -f "$test_file" ]; then
        output=$(uv run pytest "$test_file" -v --tb=short 2>&1)
        exit_code=$?
        
        # Parse results
        if [[ $output =~ ([0-9]+)\ passed ]]; then
            suite_passed="${BASH_REMATCH[1]}"
            passed_tests=$((passed_tests + suite_passed))
        fi
        
        if [[ $output =~ ([0-9]+)\ failed ]]; then
            suite_failed="${BASH_REMATCH[1]}"
            failed_tests=$((failed_tests + suite_failed))
        fi
        
        if [[ $output =~ ([0-9]+)\ skipped ]]; then
            suite_skipped="${BASH_REMATCH[1]}"
            skipped_tests=$((skipped_tests + suite_skipped))
        fi
        
        if [[ $output =~ ([0-9]+)\ error ]]; then
            suite_errors="${BASH_REMATCH[1]}"
            errors=$((errors + suite_errors))
        fi
        
        # Display result
        if [ $exit_code -eq 0 ]; then
            echo -e "${GREEN}✓ PASSED${NC}"
        else
            echo -e "${RED}✗ FAILED${NC}"
            echo "Error details:"
            echo "$output" | grep -E "(FAILED|ERROR|AssertionError)" | head -5
        fi
    else
        echo -e "${RED}✗ NOT FOUND${NC}"
        errors=$((errors + 1))
    fi
    
    echo ""
done

# Calculate totals
total_tests=$((passed_tests + failed_tests + skipped_tests))

# Generate summary report
echo "=================================================================="
echo -e "${BLUE}TEST SUMMARY REPORT${NC}"
echo "=================================================================="
echo ""
echo "Test Execution Summary:"
echo "----------------------"
echo -e "Total Tests Run:    ${total_tests}"
echo -e "Tests Passed:       ${GREEN}${passed_tests}${NC}"
echo -e "Tests Failed:       ${RED}${failed_tests}${NC}"
echo -e "Tests Skipped:      ${YELLOW}${skipped_tests}${NC}"
echo -e "Errors:             ${RED}${errors}${NC}"
echo ""

# Calculate success rate
if [ $total_tests -gt 0 ]; then
    success_rate=$(( passed_tests * 100 / total_tests ))
    echo -e "Success Rate:       ${success_rate}%"
else
    echo -e "Success Rate:       N/A (no tests run)"
fi

echo ""
echo "Test Coverage Areas:"
echo "-------------------"
echo "✓ Thought Models and Validation"
echo "✓ Shared Context and Memory Management"
echo "✓ Tool Selection Algorithm"
echo "✓ LLM Provider Configuration"
echo "✓ MCP Endpoint Integration"
echo "✓ Error Handling and Recovery"
echo "✓ Real-World Scenarios"
echo ""

# Quick checks
echo "Quick Checks:"
echo "------------"

# Check for pyright
echo -n "Type Checking (pyright): "
if uv run pyright . --pythonversion 3.10 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PASSED${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
fi

# Check for ruff
echo -n "Linting (ruff):          "
if ruff check . > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PASSED${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
fi

echo ""
echo "=================================================================="

# Exit with appropriate code
if [ $failed_tests -gt 0 ] || [ $errors -gt 0 ]; then
    echo -e "${RED}❌ TEST SUITE FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}✅ TEST SUITE PASSED${NC}"
    exit 0
fi